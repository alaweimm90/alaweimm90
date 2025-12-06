"""Finite Element Method for PDEs.
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
"""
import numpy as np
from typing import Callable, Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from scipy import sparse
from scipy.sparse import linalg as spla
from scipy.integrate import quad
import warnings
@dataclass
class FEMResult:
    """Result of FEM computation."""
    nodes: np.ndarray
    elements: np.ndarray
    solution: np.ndarray
    dofs: np.ndarray
    success: bool
    message: str
    assembly_time: float = 0.0
    solve_time: float = 0.0
    n_elements: int = 0
    n_nodes: int = 0
@dataclass
class Element:
    """Finite element definition."""
    nodes: List[int]  # Global node numbers
    coordinates: np.ndarray  # Physical coordinates
    element_type: str = "linear"
    material_id: int = 0
class FiniteElement(ABC):
    """Abstract base class for finite elements."""
    def __init__(self, element_type: str, order: int):
        """Initialize finite element.
        Args:
            element_type: Type of element ('linear', 'quadratic', etc.)
            order: Polynomial order
        """
        self.element_type = element_type
        self.order = order
        self.n_nodes = order + 1
    @abstractmethod
    def shape_functions(self, xi: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluate shape functions at reference coordinates."""
        pass
    @abstractmethod
    def shape_derivatives(self, xi: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluate shape function derivatives at reference coordinates."""
        pass
    @abstractmethod
    def quadrature_rule(self, order: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get quadrature points and weights."""
        pass
    def jacobian(self, xi: float, coordinates: np.ndarray) -> float:
        """Compute Jacobian of coordinate transformation.
        Args:
            xi: Reference coordinate
            coordinates: Physical coordinates of element nodes
        Returns:
            Jacobian determinant
        """
        dN_dxi = self.shape_derivatives(xi)
        dx_dxi = np.sum(dN_dxi * coordinates)
        return dx_dxi
    def physical_coordinates(self, xi: Union[float, np.ndarray],
                           coordinates: np.ndarray) -> Union[float, np.ndarray]:
        """Map from reference to physical coordinates.
        Args:
            xi: Reference coordinate(s)
            coordinates: Physical coordinates of element nodes
        Returns:
            Physical coordinate(s)
        """
        N = self.shape_functions(xi)
        if np.isscalar(xi):
            return np.sum(N * coordinates)
        else:
            return N @ coordinates
class LinearElement(FiniteElement):
    """Linear Lagrange element (1D)."""
    def __init__(self):
        super().__init__("linear", 1)
    def shape_functions(self, xi: Union[float, np.ndarray]) -> np.ndarray:
        """Linear shape functions N1 = (1-xi)/2, N2 = (1+xi)/2."""
        if np.isscalar(xi):
            return np.array([0.5 * (1 - xi), 0.5 * (1 + xi)])
        else:
            N = np.zeros((len(xi), 2))
            N[:, 0] = 0.5 * (1 - xi)
            N[:, 1] = 0.5 * (1 + xi)
            return N
    def shape_derivatives(self, xi: Union[float, np.ndarray]) -> np.ndarray:
        """Linear shape function derivatives dN/dxi."""
        if np.isscalar(xi):
            return np.array([-0.5, 0.5])
        else:
            dN = np.zeros((len(xi), 2))
            dN[:, 0] = -0.5
            dN[:, 1] = 0.5
            return dN
    def quadrature_rule(self, order: int) -> Tuple[np.ndarray, np.ndarray]:
        """Gauss-Legendre quadrature for interval [-1, 1]."""
        if order <= 1:
            # 1-point rule (exact for linear)
            xi = np.array([0.0])
            w = np.array([2.0])
        elif order <= 3:
            # 2-point rule (exact for cubic)
            xi = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
            w = np.array([1.0, 1.0])
        elif order <= 5:
            # 3-point rule (exact for quintic)
            xi = np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)])
            w = np.array([5/9, 8/9, 5/9])
        else:
            # Higher order rules would go here
            raise NotImplementedError(f"Quadrature order {order} not implemented")
        return xi, w
class QuadraticElement(FiniteElement):
    """Quadratic Lagrange element (1D)."""
    def __init__(self):
        super().__init__("quadratic", 2)
    def shape_functions(self, xi: Union[float, np.ndarray]) -> np.ndarray:
        """Quadratic shape functions."""
        if np.isscalar(xi):
            N1 = 0.5 * xi * (xi - 1)    # Node at xi = -1
            N2 = (1 - xi) * (1 + xi)    # Node at xi = 0
            N3 = 0.5 * xi * (xi + 1)    # Node at xi = 1
            return np.array([N1, N2, N3])
        else:
            N = np.zeros((len(xi), 3))
            N[:, 0] = 0.5 * xi * (xi - 1)
            N[:, 1] = (1 - xi) * (1 + xi)
            N[:, 2] = 0.5 * xi * (xi + 1)
            return N
    def shape_derivatives(self, xi: Union[float, np.ndarray]) -> np.ndarray:
        """Quadratic shape function derivatives."""
        if np.isscalar(xi):
            dN1 = xi - 0.5
            dN2 = -2 * xi
            dN3 = xi + 0.5
            return np.array([dN1, dN2, dN3])
        else:
            dN = np.zeros((len(xi), 3))
            dN[:, 0] = xi - 0.5
            dN[:, 1] = -2 * xi
            dN[:, 2] = xi + 0.5
            return dN
    def quadrature_rule(self, order: int) -> Tuple[np.ndarray, np.ndarray]:
        """Gauss-Legendre quadrature for quadratic elements."""
        if order <= 3:
            # 2-point rule
            xi = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
            w = np.array([1.0, 1.0])
        elif order <= 5:
            # 3-point rule (exact for quintic)
            xi = np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)])
            w = np.array([5/9, 8/9, 5/9])
        else:
            raise NotImplementedError(f"Quadrature order {order} not implemented")
        return xi, w
class FEMAssembler:
    """Finite element matrix assembler."""
    def __init__(self, mesh: np.ndarray, elements: np.ndarray,
                 element_type: str = "linear"):
        """Initialize FEM assembler.
        Args:
            mesh: Mesh node coordinates
            elements: Element connectivity
            element_type: Type of finite element
        """
        self.mesh = mesh
        self.elements = elements
        self.n_nodes = len(mesh)
        self.n_elements = len(elements)
        # Create finite element
        if element_type == "linear":
            self.fe = LinearElement()
        elif element_type == "quadratic":
            self.fe = QuadraticElement()
        else:
            raise ValueError(f"Unknown element type: {element_type}")
    def assemble_stiffness_matrix(self, coefficient: Union[float, Callable] = 1.0) -> sparse.csr_matrix:
        """Assemble global stiffness matrix.
        Args:
            coefficient: Diffusion coefficient (scalar or function)
        Returns:
            Global stiffness matrix
        """
        # Initialize global matrix
        K = sparse.lil_matrix((self.n_nodes, self.n_nodes))
        # Quadrature rule
        xi_q, w_q = self.fe.quadrature_rule(self.fe.order * 2)
        for e in range(self.n_elements):
            # Element nodes and coordinates
            element_nodes = self.elements[e]
            element_coords = self.mesh[element_nodes]
            # Element stiffness matrix
            K_e = np.zeros((self.fe.n_nodes, self.fe.n_nodes))
            # Numerical integration
            for q in range(len(xi_q)):
                xi = xi_q[q]
                w = w_q[q]
                # Shape function derivatives in reference coordinates
                dN_dxi = self.fe.shape_derivatives(xi)
                # Jacobian
                J = self.fe.jacobian(xi, element_coords)
                # Shape function derivatives in physical coordinates
                dN_dx = dN_dxi / J
                # Coefficient value
                if callable(coefficient):
                    x = self.fe.physical_coordinates(xi, element_coords)
                    coeff_val = coefficient(x)
                else:
                    coeff_val = coefficient
                # Add to element matrix
                K_e += coeff_val * np.outer(dN_dx, dN_dx) * J * w
            # Assemble into global matrix
            for i in range(self.fe.n_nodes):
                for j in range(self.fe.n_nodes):
                    K[element_nodes[i], element_nodes[j]] += K_e[i, j]
        return K.tocsr()
    def assemble_mass_matrix(self, density: Union[float, Callable] = 1.0) -> sparse.csr_matrix:
        """Assemble global mass matrix.
        Args:
            density: Density coefficient (scalar or function)
        Returns:
            Global mass matrix
        """
        # Initialize global matrix
        M = sparse.lil_matrix((self.n_nodes, self.n_nodes))
        # Quadrature rule
        xi_q, w_q = self.fe.quadrature_rule(self.fe.order * 2)
        for e in range(self.n_elements):
            # Element nodes and coordinates
            element_nodes = self.elements[e]
            element_coords = self.mesh[element_nodes]
            # Element mass matrix
            M_e = np.zeros((self.fe.n_nodes, self.fe.n_nodes))
            # Numerical integration
            for q in range(len(xi_q)):
                xi = xi_q[q]
                w = w_q[q]
                # Shape functions
                N = self.fe.shape_functions(xi)
                # Jacobian
                J = self.fe.jacobian(xi, element_coords)
                # Density value
                if callable(density):
                    x = self.fe.physical_coordinates(xi, element_coords)
                    rho_val = density(x)
                else:
                    rho_val = density
                # Add to element matrix
                M_e += rho_val * np.outer(N, N) * J * w
            # Assemble into global matrix
            for i in range(self.fe.n_nodes):
                for j in range(self.fe.n_nodes):
                    M[element_nodes[i], element_nodes[j]] += M_e[i, j]
        return M.tocsr()
    def assemble_load_vector(self, source: Union[float, Callable] = 0.0) -> np.ndarray:
        """Assemble global load vector.
        Args:
            source: Source term (scalar or function)
        Returns:
            Global load vector
        """
        # Initialize global vector
        f = np.zeros(self.n_nodes)
        # Quadrature rule
        xi_q, w_q = self.fe.quadrature_rule(self.fe.order + 1)
        for e in range(self.n_elements):
            # Element nodes and coordinates
            element_nodes = self.elements[e]
            element_coords = self.mesh[element_nodes]
            # Element load vector
            f_e = np.zeros(self.fe.n_nodes)
            # Numerical integration
            for q in range(len(xi_q)):
                xi = xi_q[q]
                w = w_q[q]
                # Shape functions
                N = self.fe.shape_functions(xi)
                # Jacobian
                J = self.fe.jacobian(xi, element_coords)
                # Source value
                if callable(source):
                    x = self.fe.physical_coordinates(xi, element_coords)
                    source_val = source(x)
                else:
                    source_val = source
                # Add to element vector
                f_e += source_val * N * J * w
            # Assemble into global vector
            for i in range(self.fe.n_nodes):
                f[element_nodes[i]] += f_e[i]
        return f
class FEMSolver:
    """General finite element solver."""
    def __init__(self, mesh: np.ndarray, elements: np.ndarray,
                 boundary_conditions: Dict[str, Any],
                 element_type: str = "linear"):
        """Initialize FEM solver.
        Args:
            mesh: Mesh node coordinates
            elements: Element connectivity
            boundary_conditions: Boundary conditions
            element_type: Type of finite element
        """
        self.mesh = mesh
        self.elements = elements
        self.boundary_conditions = boundary_conditions
        self.element_type = element_type
        self.assembler = FEMAssembler(mesh, elements, element_type)
        self.n_nodes = len(mesh)
    def apply_boundary_conditions(self, K: sparse.csr_matrix,
                                 f: np.ndarray) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """Apply boundary conditions to system.
        Args:
            K: Stiffness matrix
            f: Load vector
        Returns:
            Modified matrix and vector
        """
        K_mod = K.copy()
        f_mod = f.copy()
        # Apply Dirichlet boundary conditions
        if 'dirichlet' in self.boundary_conditions:
            dirichlet_bc = self.boundary_conditions['dirichlet']
            for node, value in dirichlet_bc.items():
                if isinstance(node, int) and 0 <= node < self.n_nodes:
                    # Set row to identity
                    K_mod.data[K_mod.indptr[node]:K_mod.indptr[node+1]] = 0
                    K_mod[node, node] = 1.0
                    f_mod[node] = value
        # Apply Neumann boundary conditions (natural, included in load vector)
        if 'neumann' in self.boundary_conditions:
            neumann_bc = self.boundary_conditions['neumann']
            for node, flux in neumann_bc.items():
                if isinstance(node, int) and 0 <= node < self.n_nodes:
                    # Add flux contribution
                    f_mod[node] += flux
        return K_mod, f_mod
    def solve_poisson(self, source: Union[float, Callable] = 0.0,
                     diffusivity: Union[float, Callable] = 1.0) -> FEMResult:
        """Solve Poisson equation -∇·(k∇u) = f.
        Args:
            source: Source term
            diffusivity: Diffusion coefficient
        Returns:
            FEMResult with solution
        """
        import time
        start_time = time.time()
        # Assemble matrices
        assembly_start = time.time()
        K = self.assembler.assemble_stiffness_matrix(diffusivity)
        f = self.assembler.assemble_load_vector(source)
        assembly_time = time.time() - assembly_start
        # Apply boundary conditions
        K, f = self.apply_boundary_conditions(K, f)
        # Solve system
        solve_start = time.time()
        try:
            u = spla.spsolve(K, f)
            success = True
            message = "Poisson equation solved successfully"
        except Exception as e:
            u = np.zeros(self.n_nodes)
            success = False
            message = f"FEM solve failed: {e}"
        solve_time = time.time() - solve_start
        return FEMResult(
            nodes=self.mesh,
            elements=self.elements,
            solution=u,
            dofs=np.arange(self.n_nodes),
            success=success,
            message=message,
            assembly_time=assembly_time,
            solve_time=solve_time,
            n_elements=len(self.elements),
            n_nodes=self.n_nodes
        )
    def solve_heat_equation(self, initial_condition: Callable,
                           time_span: Tuple[float, float],
                           dt: float,
                           thermal_diffusivity: Union[float, Callable] = 1.0,
                           source: Union[float, Callable] = 0.0) -> FEMResult:
        """Solve transient heat equation using FEM.
        Args:
            initial_condition: Initial condition function
            time_span: Time interval
            dt: Time step
            thermal_diffusivity: Thermal diffusivity
            source: Source term
        Returns:
            FEMResult with time-dependent solution
        """
        t0, tf = time_span
        nt = int((tf - t0) / dt) + 1
        t = np.linspace(t0, tf, nt)
        # Assemble matrices
        K = self.assembler.assemble_stiffness_matrix(thermal_diffusivity)
        M = self.assembler.assemble_mass_matrix()
        # Time-dependent solution storage
        u_time = np.zeros((nt, self.n_nodes))
        # Initial condition
        u_time[0] = initial_condition(self.mesh)
        # Time stepping (implicit Euler)
        for n in range(nt - 1):
            # Load vector
            if callable(source):
                f = self.assembler.assemble_load_vector(
                    lambda x: source(x, t[n+1]))
            else:
                f = self.assembler.assemble_load_vector(source)
            # System matrix: (M + dt*K)*u^{n+1} = M*u^n + dt*f
            A = M + dt * K
            b = M @ u_time[n] + dt * f
            # Apply boundary conditions
            A, b = self.apply_boundary_conditions(A, b)
            # Solve
            u_time[n+1] = spla.spsolve(A, b)
        return FEMResult(
            nodes=self.mesh,
            elements=self.elements,
            solution=u_time,
            dofs=np.arange(self.n_nodes),
            success=True,
            message="Heat equation solved with FEM",
            n_elements=len(self.elements),
            n_nodes=self.n_nodes
        )
# Utility functions
def create_1d_mesh(domain: Tuple[float, float], n_elements: int,
                   element_type: str = "linear") -> Tuple[np.ndarray, np.ndarray]:
    """Create 1D finite element mesh.
    Args:
        domain: Domain interval (a, b)
        n_elements: Number of elements
        element_type: Type of elements
    Returns:
        Tuple of (nodes, elements)
    """
    a, b = domain
    if element_type == "linear":
        # Linear elements: 2 nodes per element
        n_nodes = n_elements + 1
        nodes = np.linspace(a, b, n_nodes)
        elements = np.zeros((n_elements, 2), dtype=int)
        for i in range(n_elements):
            elements[i] = [i, i + 1]
    elif element_type == "quadratic":
        # Quadratic elements: 3 nodes per element (with shared nodes)
        n_nodes = 2 * n_elements + 1
        nodes = np.linspace(a, b, n_nodes)
        elements = np.zeros((n_elements, 3), dtype=int)
        for i in range(n_elements):
            elements[i] = [2*i, 2*i + 1, 2*i + 2]
    else:
        raise ValueError(f"Unknown element type: {element_type}")
    return nodes, elements
def assemble_stiffness_matrix(mesh: np.ndarray, elements: np.ndarray,
                             coefficient: Union[float, Callable] = 1.0,
                             element_type: str = "linear") -> sparse.csr_matrix:
    """Convenience function to assemble stiffness matrix.
    Args:
        mesh: Mesh nodes
        elements: Element connectivity
        coefficient: Diffusion coefficient
        element_type: Type of finite element
    Returns:
        Global stiffness matrix
    """
    assembler = FEMAssembler(mesh, elements, element_type)
    return assembler.assemble_stiffness_matrix(coefficient)
def assemble_mass_matrix(mesh: np.ndarray, elements: np.ndarray,
                        density: Union[float, Callable] = 1.0,
                        element_type: str = "linear") -> sparse.csr_matrix:
    """Convenience function to assemble mass matrix.
    Args:
        mesh: Mesh nodes
        elements: Element connectivity
        density: Density coefficient
        element_type: Type of finite element
    Returns:
        Global mass matrix
    """
    assembler = FEMAssembler(mesh, elements, element_type)
    return assembler.assemble_mass_matrix(density)
def solve_fem_poisson(domain: Tuple[float, float], n_elements: int,
                     boundary_conditions: Dict[str, Any],
                     source: Union[float, Callable] = 0.0,
                     diffusivity: Union[float, Callable] = 1.0,
                     element_type: str = "linear") -> FEMResult:
    """Solve Poisson equation using FEM.
    Args:
        domain: Spatial domain
        n_elements: Number of finite elements
        boundary_conditions: Boundary conditions
        source: Source term
        diffusivity: Diffusion coefficient
        element_type: Type of finite element
    Returns:
        FEMResult with solution
    """
    # Create mesh
    mesh, elements = create_1d_mesh(domain, n_elements, element_type)
    # Create solver
    solver = FEMSolver(mesh, elements, boundary_conditions, element_type)
    # Solve
    return solver.solve_poisson(source, diffusivity)
def compute_l2_error(u_exact: Callable, u_fem: np.ndarray,
                    mesh: np.ndarray, elements: np.ndarray,
                    element_type: str = "linear") -> float:
    """Compute L2 error between exact and FEM solution.
    Args:
        u_exact: Exact solution function
        u_fem: FEM solution
        mesh: Mesh nodes
        elements: Element connectivity
        element_type: Type of finite element
    Returns:
        L2 error norm
    """
    # Create finite element
    if element_type == "linear":
        fe = LinearElement()
    elif element_type == "quadratic":
        fe = QuadraticElement()
    else:
        raise ValueError(f"Unknown element type: {element_type}")
    # Quadrature rule
    xi_q, w_q = fe.quadrature_rule(fe.order * 2)
    error_squared = 0.0
    for e in range(len(elements)):
        element_nodes = elements[e]
        element_coords = mesh[element_nodes]
        u_e = u_fem[element_nodes]
        # Integrate error over element
        for q in range(len(xi_q)):
            xi = xi_q[q]
            w = w_q[q]
            # Physical coordinate
            x = fe.physical_coordinates(xi, element_coords)
            # FEM solution at quadrature point
            N = fe.shape_functions(xi)
            u_fem_q = np.sum(N * u_e)
            # Exact solution at quadrature point
            u_exact_q = u_exact(x)
            # Jacobian
            J = fe.jacobian(xi, element_coords)
            # Add to error
            error_squared += (u_fem_q - u_exact_q)**2 * J * w
    return np.sqrt(error_squared)
def fem_convergence_study(domain: Tuple[float, float],
                         u_exact: Callable,
                         boundary_conditions: Dict[str, Any],
                         source: Union[float, Callable],
                         element_counts: List[int],
                         element_type: str = "linear") -> Dict[str, Any]:
    """Perform FEM convergence study.
    Args:
        domain: Spatial domain
        u_exact: Exact solution function
        boundary_conditions: Boundary conditions
        source: Source term
        element_counts: List of element counts to test
        element_type: Type of finite element
    Returns:
        Dictionary with convergence results
    """
    errors = []
    h_values = []
    for n_elements in element_counts:
        # Solve with n_elements
        result = solve_fem_poisson(domain, n_elements, boundary_conditions,
                                  source, element_type=element_type)
        if result.success:
            # Compute error
            error = compute_l2_error(u_exact, result.solution,
                                   result.nodes, result.elements, element_type)
            errors.append(error)
            # Element size
            h = (domain[1] - domain[0]) / n_elements
            h_values.append(h)
        else:
            warnings.warn(f"FEM solve failed for {n_elements} elements")
    # Estimate convergence rate
    if len(errors) >= 2:
        log_errors = np.log(errors)
        log_h = np.log(h_values)
        # Linear regression
        A = np.vstack([log_h, np.ones(len(log_h))]).T
        rate, _ = np.linalg.lstsq(A, log_errors, rcond=None)[0]
    else:
        rate = 0.0
    return {
        'element_counts': element_counts,
        'h_values': h_values,
        'errors': errors,
        'convergence_rate': rate,
        'element_type': element_type
    }