"""Utility Functions for ODE and PDE Solvers.
This module provides utility functions for mesh generation, derivative computation,
interpolation, convergence analysis, and other common operations for ODE/PDE solving.
Classes:
    MeshGenerator: Mesh generation utilities
    DerivativeOperator: Finite difference derivative operators
Functions:
    generate_uniform_mesh: Create uniform meshes
    generate_adaptive_mesh: Create adaptive meshes
    compute_derivatives: Compute numerical derivatives
    interpolate_solution: Interpolate solutions
    check_convergence: Check numerical convergence
    estimate_error: Estimate numerical errors
Author: Berkeley SciComp Team
Date: 2024
"""
import numpy as np
from typing import Callable, Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy.interpolate import interp1d, griddata, UnivariateSpline
from scipy.sparse import diags, eye, csr_matrix
import warnings
@dataclass
class MeshInfo:
    """Information about generated mesh."""
    nodes: np.ndarray
    elements: Optional[np.ndarray] = None
    boundaries: Optional[Dict[str, List[int]]] = None
    spacing: Optional[Union[float, np.ndarray]] = None
    dimension: int = 1
    n_nodes: int = 0
    def __post_init__(self):
        """Post-initialization setup."""
        self.n_nodes = len(self.nodes)
        if self.nodes.ndim > 1:
            self.dimension = self.nodes.shape[1]
class MeshGenerator:
    """Mesh generation utilities for ODE and PDE problems."""
    def __init__(self):
        """Initialize mesh generator."""
        self.tolerance = 1e-8
    def uniform_1d(self, domain: Tuple[float, float], n_points: int) -> MeshInfo:
        """Generate uniform 1D mesh.
        Args:
            domain: Domain interval (a, b)
            n_points: Number of grid points
        Returns:
            MeshInfo object
        """
        a, b = domain
        nodes = np.linspace(a, b, n_points)
        spacing = (b - a) / (n_points - 1)
        # Boundary information
        boundaries = {
            'left': [0],
            'right': [n_points - 1]
        }
        return MeshInfo(
            nodes=nodes,
            boundaries=boundaries,
            spacing=spacing,
            dimension=1,
            n_nodes=n_points
        )
    def uniform_2d(self, domain: Dict[str, Tuple[float, float]],
                   n_points: Dict[str, int]) -> MeshInfo:
        """Generate uniform 2D mesh.
        Args:
            domain: Domain specification {'x': (x0, x1), 'y': (y0, y1)}
            n_points: Number of points {'x': nx, 'y': ny}
        Returns:
            MeshInfo object
        """
        x_domain = domain['x']
        y_domain = domain['y']
        nx = n_points['x']
        ny = n_points['y']
        # Create 1D grids
        x = np.linspace(x_domain[0], x_domain[1], nx)
        y = np.linspace(y_domain[0], y_domain[1], ny)
        # Create 2D mesh
        X, Y = np.meshgrid(x, y, indexing='ij')
        nodes = np.column_stack([X.ravel(), Y.ravel()])
        # Spacing
        dx = (x_domain[1] - x_domain[0]) / (nx - 1)
        dy = (y_domain[1] - y_domain[0]) / (ny - 1)
        spacing = np.array([dx, dy])
        # Boundary information
        boundaries = {
            'left': list(range(0, nx * ny, nx)),  # x = x0
            'right': list(range(nx - 1, nx * ny, nx)),  # x = x1
            'bottom': list(range(nx)),  # y = y0
            'top': list(range(nx * (ny - 1), nx * ny))  # y = y1
        }
        return MeshInfo(
            nodes=nodes,
            boundaries=boundaries,
            spacing=spacing,
            dimension=2,
            n_nodes=nx * ny
        )
    def adaptive_1d(self, domain: Tuple[float, float],
                   refinement_function: Callable,
                   max_points: int = 1000,
                   min_spacing: float = 1e-6) -> MeshInfo:
        """Generate adaptive 1D mesh based on refinement criterion.
        Args:
            domain: Domain interval (a, b)
            refinement_function: Function to determine refinement need
            max_points: Maximum number of points
            min_spacing: Minimum allowed spacing
        Returns:
            MeshInfo object
        """
        a, b = domain
        # Start with coarse uniform mesh
        nodes = [a, b]
        # Adaptive refinement
        while len(nodes) < max_points:
            refined = False
            # Check each interval for refinement
            new_nodes = []
            for i in range(len(nodes) - 1):
                new_nodes.append(nodes[i])
                x_left = nodes[i]
                x_right = nodes[i + 1]
                x_mid = (x_left + x_right) / 2
                # Check if refinement is needed
                if (x_right - x_left > min_spacing and
                    refinement_function(x_left, x_right, x_mid)):
                    new_nodes.append(x_mid)
                    refined = True
            new_nodes.append(nodes[-1])
            nodes = new_nodes
            if not refined:
                break
        nodes = np.array(sorted(nodes))
        # Compute spacing (variable)
        spacing = np.diff(nodes)
        # Boundary information
        boundaries = {
            'left': [0],
            'right': [len(nodes) - 1]
        }
        return MeshInfo(
            nodes=nodes,
            boundaries=boundaries,
            spacing=spacing,
            dimension=1,
            n_nodes=len(nodes)
        )
    def graded_1d(self, domain: Tuple[float, float], n_points: int,
                 grading_factor: float = 1.0,
                 boundary: str = 'left') -> MeshInfo:
        """Generate graded 1D mesh with clustering near boundary.
        Args:
            domain: Domain interval (a, b)
            n_points: Number of grid points
            grading_factor: Grading factor (1.0 = uniform, > 1 = clustered)
            boundary: Which boundary to cluster near ('left', 'right', 'both')
        Returns:
            MeshInfo object
        """
        a, b = domain
        if grading_factor == 1.0:
            # Uniform mesh
            return self.uniform_1d(domain, n_points)
        # Generate graded mesh
        xi = np.linspace(0, 1, n_points)
        if boundary == 'left':
            # Cluster near left boundary
            eta = (1 - np.exp(-grading_factor * xi)) / (1 - np.exp(-grading_factor))
        elif boundary == 'right':
            # Cluster near right boundary
            eta = np.exp(grading_factor * xi) - 1 / (np.exp(grading_factor) - 1)
        elif boundary == 'both':
            # Cluster near both boundaries (sinusoidal)
            eta = 0.5 * (1 - np.cos(np.pi * xi))
        else:
            raise ValueError(f"Unknown boundary specification: {boundary}")
        # Map to physical domain
        nodes = a + (b - a) * eta
        # Spacing (variable)
        spacing = np.diff(nodes)
        # Boundary information
        boundaries = {
            'left': [0],
            'right': [n_points - 1]
        }
        return MeshInfo(
            nodes=nodes,
            boundaries=boundaries,
            spacing=spacing,
            dimension=1,
            n_nodes=n_points
        )
class DerivativeOperator:
    """Finite difference derivative operators."""
    def __init__(self, order: int = 2, stencil_width: int = 3):
        """Initialize derivative operator.
        Args:
            order: Order of accuracy
            stencil_width: Width of finite difference stencil
        """
        self.order = order
        self.stencil_width = stencil_width
    def first_derivative_1d(self, n: int, spacing: float,
                           boundary_order: int = 1) -> csr_matrix:
        """Build first derivative matrix for 1D.
        Args:
            n: Number of grid points
            spacing: Grid spacing
            boundary_order: Order of accuracy at boundaries
        Returns:
            Sparse matrix for first derivative
        """
        if self.order == 2:
            # Central differences
            diagonals = [-1, 0, 1]
            data = [
                -np.ones(n-1),  # Lower diagonal
                np.zeros(n),    # Main diagonal
                np.ones(n-1)    # Upper diagonal
            ]
            # Handle boundaries
            if boundary_order == 1:
                # First-order forward/backward differences
                data[1][0] = -1    # du/dx[0] = (-u[0] + u[1])/h
                data[2][0] = 1
                data[0][-1] = -1   # du/dx[-1] = (-u[-2] + u[-1])/h
                data[1][-1] = 1
            D = diags(data, diagonals, shape=(n, n), format='csr')
            D = D / (2 * spacing)
            # Fix boundaries
            if boundary_order == 1:
                D[0, :] = 0
                D[0, 0] = -1/spacing
                D[0, 1] = 1/spacing
                D[-1, :] = 0
                D[-1, -2] = -1/spacing
                D[-1, -1] = 1/spacing
        else:
            raise NotImplementedError(f"Order {self.order} not implemented")
        return D
    def second_derivative_1d(self, n: int, spacing: float) -> csr_matrix:
        """Build second derivative matrix for 1D.
        Args:
            n: Number of grid points
            spacing: Grid spacing
        Returns:
            Sparse matrix for second derivative
        """
        # Central differences: d²u/dx² ≈ (u[i-1] - 2u[i] + u[i+1])/h²
        data = [
            np.ones(n-1),   # Lower diagonal
            -2*np.ones(n),  # Main diagonal
            np.ones(n-1)    # Upper diagonal
        ]
        D2 = diags(data, [-1, 0, 1], shape=(n, n), format='csr')
        D2 = D2 / (spacing**2)
        return D2
    def laplacian_2d(self, nx: int, ny: int,
                    dx: float, dy: float) -> csr_matrix:
        """Build 2D Laplacian matrix.
        Args:
            nx: Number of points in x direction
            ny: Number of points in y direction
            dx: Spacing in x direction
            dy: Spacing in y direction
        Returns:
            Sparse matrix for 2D Laplacian
        """
        # Build 1D second derivative operators
        D2x = self.second_derivative_1d(nx, dx)
        D2y = self.second_derivative_1d(ny, dy)
        # 2D Laplacian via Kronecker sum
        Ix = eye(nx)
        Iy = eye(ny)
        from scipy.sparse import kron
        L = kron(Iy, D2x) + kron(D2y, Ix)
        return L
# Mesh generation functions
def generate_uniform_mesh(domain: Union[Tuple[float, float], Dict[str, Tuple[float, float]]],
                         n_points: Union[int, Dict[str, int]]) -> MeshInfo:
    """Generate uniform mesh.
    Args:
        domain: Domain specification
        n_points: Number of points
    Returns:
        MeshInfo object
    """
    generator = MeshGenerator()
    if isinstance(domain, tuple):
        # 1D case
        return generator.uniform_1d(domain, n_points)
    else:
        # 2D case
        return generator.uniform_2d(domain, n_points)
def generate_adaptive_mesh(domain: Tuple[float, float],
                          refinement_function: Callable,
                          max_points: int = 1000) -> MeshInfo:
    """Generate adaptive mesh.
    Args:
        domain: 1D domain interval
        refinement_function: Refinement criterion function
        max_points: Maximum number of points
    Returns:
        MeshInfo object
    """
    generator = MeshGenerator()
    return generator.adaptive_1d(domain, refinement_function, max_points)
def compute_derivatives(u: np.ndarray, spacing: Union[float, np.ndarray],
                       derivative_order: int = 1,
                       method: str = 'central') -> np.ndarray:
    """Compute numerical derivatives.
    Args:
        u: Function values
        spacing: Grid spacing
        derivative_order: Order of derivative (1 or 2)
        method: Finite difference method ('forward', 'backward', 'central')
    Returns:
        Derivative values
    """
    n = len(u)
    if derivative_order == 1:
        if method == 'central':
            # Central differences
            du = np.zeros_like(u)
            # Interior points
            du[1:-1] = (u[2:] - u[:-2]) / (2 * spacing)
            # Boundaries (forward/backward differences)
            du[0] = (-3*u[0] + 4*u[1] - u[2]) / (2 * spacing)
            du[-1] = (3*u[-1] - 4*u[-2] + u[-3]) / (2 * spacing)
        elif method == 'forward':
            du = np.zeros_like(u)
            du[:-1] = (u[1:] - u[:-1]) / spacing
            du[-1] = du[-2]  # Extrapolate
        elif method == 'backward':
            du = np.zeros_like(u)
            du[1:] = (u[1:] - u[:-1]) / spacing
            du[0] = du[1]  # Extrapolate
        else:
            raise ValueError(f"Unknown method: {method}")
    elif derivative_order == 2:
        # Second derivative (central differences)
        d2u = np.zeros_like(u)
        # Interior points
        d2u[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / (spacing**2)
        # Boundaries (second-order accurate)
        d2u[0] = (2*u[0] - 5*u[1] + 4*u[2] - u[3]) / (spacing**2)
        d2u[-1] = (2*u[-1] - 5*u[-2] + 4*u[-3] - u[-4]) / (spacing**2)
        return d2u
    else:
        raise ValueError(f"Derivative order {derivative_order} not supported")
    return du
def interpolate_solution(x_old: np.ndarray, u_old: np.ndarray,
                        x_new: np.ndarray,
                        method: str = 'linear') -> np.ndarray:
    """Interpolate solution to new grid.
    Args:
        x_old: Old grid points
        u_old: Solution values on old grid
        x_new: New grid points
        method: Interpolation method ('linear', 'cubic', 'spline')
    Returns:
        Interpolated solution on new grid
    """
    if method == 'linear':
        f = interp1d(x_old, u_old, kind='linear',
                    bounds_error=False, fill_value='extrapolate')
    elif method == 'cubic':
        f = interp1d(x_old, u_old, kind='cubic',
                    bounds_error=False, fill_value='extrapolate')
    elif method == 'spline':
        f = UnivariateSpline(x_old, u_old, s=0)  # s=0 for interpolation
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    return f(x_new)
def check_convergence(residuals: List[float],
                     tolerance: float = 1e-6,
                     min_iterations: int = 3) -> Tuple[bool, Dict[str, Any]]:
    """Check convergence of iterative method.
    Args:
        residuals: History of residual values
        tolerance: Convergence tolerance
        min_iterations: Minimum number of iterations
    Returns:
        Tuple of (converged, info_dict)
    """
    if len(residuals) < min_iterations:
        return False, {'reason': 'insufficient_iterations'}
    current_residual = residuals[-1]
    # Check absolute convergence
    if current_residual < tolerance:
        return True, {
            'reason': 'absolute_tolerance',
            'final_residual': current_residual,
            'iterations': len(residuals)
        }
    # Check relative convergence
    if len(residuals) > 1:
        relative_change = abs(residuals[-1] - residuals[-2]) / abs(residuals[-2])
        if relative_change < tolerance:
            return True, {
                'reason': 'relative_tolerance',
                'relative_change': relative_change,
                'iterations': len(residuals)
            }
    # Check stagnation
    if len(residuals) >= 10:
        recent_residuals = residuals[-5:]
        stagnation = max(recent_residuals) - min(recent_residuals)
        if stagnation < tolerance * abs(recent_residuals[-1]):
            return True, {
                'reason': 'stagnation',
                'stagnation_level': stagnation,
                'iterations': len(residuals)
            }
    return False, {'reason': 'not_converged'}
def estimate_error(u_coarse: np.ndarray, u_fine: np.ndarray,
                  refinement_ratio: int = 2,
                  expected_order: int = 2) -> Dict[str, float]:
    """Estimate numerical error using Richardson extrapolation.
    Args:
        u_coarse: Solution on coarse grid
        u_fine: Solution on fine grid (subsampled to match coarse)
        refinement_ratio: Grid refinement ratio
        expected_order: Expected order of accuracy
    Returns:
        Dictionary with error estimates
    """
    # Ensure same size (subsample fine grid)
    if len(u_fine) != len(u_coarse):
        indices = np.linspace(0, len(u_fine)-1, len(u_coarse), dtype=int)
        u_fine_sub = u_fine[indices]
    else:
        u_fine_sub = u_fine
    # Richardson extrapolation error estimate
    # E ≈ (u_fine - u_coarse) / (r^p - 1)
    # where r is refinement ratio, p is order
    error_diff = u_fine_sub - u_coarse
    richardson_factor = refinement_ratio**expected_order - 1
    error_estimate = error_diff / richardson_factor
    # Error norms
    max_error = np.max(np.abs(error_estimate))
    l2_error = np.sqrt(np.mean(error_estimate**2))
    l1_error = np.mean(np.abs(error_estimate))
    # Convergence rate estimate
    if np.max(np.abs(error_diff)) > 0:
        convergence_rate = np.log(np.max(np.abs(error_diff))) / np.log(refinement_ratio)
    else:
        convergence_rate = float('inf')  # Perfect convergence
    return {
        'max_error': max_error,
        'l2_error': l2_error,
        'l1_error': l1_error,
        'richardson_estimate': error_estimate,
        'convergence_rate': convergence_rate,
        'expected_order': expected_order
    }
def compute_cfl_condition(velocity: Union[float, np.ndarray],
                         dx: float, dt: float) -> float:
    """Compute CFL number for stability analysis.
    Args:
        velocity: Velocity (scalar or array)
        dx: Spatial grid spacing
        dt: Time step
    Returns:
        CFL number
    """
    if np.isscalar(velocity):
        max_velocity = abs(velocity)
    else:
        max_velocity = np.max(np.abs(velocity))
    return max_velocity * dt / dx
def analyze_spectral_properties(matrix: csr_matrix) -> Dict[str, Any]:
    """Analyze spectral properties of discretization matrix.
    Args:
        matrix: Discretization matrix
    Returns:
        Dictionary with spectral properties
    """
    try:
        from scipy.sparse.linalg import eigs
        # Compute a few eigenvalues
        n_eigs = min(10, matrix.shape[0] - 1)
        eigenvalues = eigs(matrix, k=n_eigs, return_eigenvectors=False)
        # Real and imaginary parts
        real_parts = np.real(eigenvalues)
        imag_parts = np.imag(eigenvalues)
        return {
            'eigenvalues': eigenvalues,
            'max_real_eigenvalue': np.max(real_parts),
            'min_real_eigenvalue': np.min(real_parts),
            'spectral_radius': np.max(np.abs(eigenvalues)),
            'condition_estimate': np.max(real_parts) / np.min(real_parts) if np.min(real_parts) > 0 else float('inf'),
            'has_complex_eigenvalues': np.any(np.abs(imag_parts) > 1e-12)
        }
    except Exception as e:
        warnings.warn(f"Spectral analysis failed: {e}")
        return {'error': str(e)}
def validate_solution(u: np.ndarray, domain_bounds: Optional[Tuple[float, float]] = None,
                     physical_bounds: Optional[Tuple[float, float]] = None) -> Dict[str, bool]:
    """Validate numerical solution for common issues.
    Args:
        u: Solution array
        domain_bounds: Expected domain bounds
        physical_bounds: Physical bounds (min, max) for solution
    Returns:
        Dictionary with validation results
    """
    validation = {
        'finite_values': np.all(np.isfinite(u)),
        'no_nans': not np.any(np.isnan(u)),
        'no_infs': not np.any(np.isinf(u)),
        'monotonic': np.all(np.diff(u) >= 0) or np.all(np.diff(u) <= 0),
        'smooth': True  # Placeholder for smoothness check
    }
    # Check physical bounds
    if physical_bounds is not None:
        min_bound, max_bound = physical_bounds
        validation['within_physical_bounds'] = (np.all(u >= min_bound) and
                                              np.all(u <= max_bound))
    # Check for oscillations (simple heuristic)
    if len(u) > 2:
        second_diff = np.diff(u, 2)
        max_oscillation = np.max(np.abs(second_diff))
        mean_value = np.mean(np.abs(u))
        validation['low_oscillation'] = max_oscillation < 0.1 * mean_value
    return validation