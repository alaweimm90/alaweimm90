"""Multiphysics Utility Functions.
This module provides utility functions for multiphysics applications including
field interpolation, data transfer, conservation checking, and convergence
diagnostics.
Functions:
    interpolate_field: Interpolate fields between meshes
    project_field: Project fields between function spaces
    check_conservation: Verify conservation laws
    compute_interface_normal: Compute interface normals
    transfer_data: Transfer data between domains
Classes:
    FieldInterpolator: Advanced field interpolation
    ConservationChecker: Conservation law verification
    ConvergenceDiagnostics: Convergence analysis tools
Author: Berkeley SciComp Team
Date: 2024
"""
import numpy as np
from typing import Dict, Tuple, Optional, Callable, Any, List, Union
from dataclasses import dataclass
from scipy import sparse
from scipy.spatial import KDTree, distance_matrix
from scipy.interpolate import griddata, RBFInterpolator
import warnings
from .coupling import CoupledSystem, CouplingInterface
@dataclass
class InterpolationParams:
    """Parameters for field interpolation."""
    method: str = "linear"  # linear, cubic, rbf
    extrapolation: str = "nearest"  # nearest, constant, linear
    smoothing: float = 0.0
    rbf_function: str = "thin_plate_spline"  # for RBF interpolation
@dataclass
class ConservationResult:
    """Result of conservation checking."""
    conserved: bool
    relative_error: float
    absolute_error: float
    conservation_type: str
    details: Dict[str, Any]
class FieldInterpolator:
    """Advanced field interpolation between meshes.
    Provides various interpolation methods for transferring
    fields between different computational domains.
    """
    def __init__(self, params: InterpolationParams = InterpolationParams()):
        """Initialize field interpolator.
        Args:
            params: Interpolation parameters
        """
        self.params = params
        self.source_tree = None
        self.interpolator = None
    def setup_interpolation(self,
                          source_mesh: Dict[str, np.ndarray],
                          target_mesh: Dict[str, np.ndarray]):
        """Setup interpolation between source and target meshes.
        Args:
            source_mesh: Source mesh data
            target_mesh: Target mesh data
        """
        self.source_nodes = source_mesh['nodes']
        self.target_nodes = target_mesh['nodes']
        # Build spatial search tree
        self.source_tree = KDTree(self.source_nodes)
        # Pre-compute interpolation weights for efficiency
        if self.params.method == "rbf":
            # Setup RBF interpolator (will be completed when data is provided)
            pass
        elif self.params.method in ["linear", "cubic"]:
            # For scipy.interpolate.griddata
            pass
    def interpolate_scalar_field(self,
                               source_field: np.ndarray,
                               fill_value: Optional[float] = None) -> np.ndarray:
        """Interpolate scalar field from source to target mesh.
        Args:
            source_field: Field values at source nodes
            fill_value: Value for extrapolation regions
        Returns:
            Interpolated field at target nodes
        """
        if self.source_nodes is None:
            raise ValueError("Must call setup_interpolation first")
        if len(source_field) != len(self.source_nodes):
            raise ValueError("Source field size mismatch")
        if fill_value is None:
            fill_value = np.mean(source_field)
        if self.params.method == "rbf":
            # Radial basis function interpolation
            rbf = RBFInterpolator(
                self.source_nodes,
                source_field,
                kernel=self.params.rbf_function,
                smoothing=self.params.smoothing
            )
            target_field = rbf(self.target_nodes)
        elif self.params.method in ["linear", "cubic"]:
            # Scipy griddata interpolation
            target_field = griddata(
                self.source_nodes,
                source_field,
                self.target_nodes,
                method=self.params.method,
                fill_value=fill_value
            )
        elif self.params.method == "nearest":
            # Nearest neighbor interpolation
            _, indices = self.source_tree.query(self.target_nodes)
            target_field = source_field[indices]
        else:
            raise ValueError(f"Unknown interpolation method: {self.params.method}")
        return target_field
    def interpolate_vector_field(self,
                               source_field: np.ndarray,
                               fill_value: Optional[float] = None) -> np.ndarray:
        """Interpolate vector field from source to target mesh.
        Args:
            source_field: Vector field at source nodes
            fill_value: Value for extrapolation
        Returns:
            Interpolated vector field at target nodes
        """
        if source_field.ndim != 2:
            raise ValueError("Vector field must be 2D array")
        n_components = source_field.shape[1]
        target_field = np.zeros((len(self.target_nodes), n_components))
        # Interpolate each component separately
        for i in range(n_components):
            target_field[:, i] = self.interpolate_scalar_field(
                source_field[:, i], fill_value
            )
        return target_field
    def compute_interpolation_error(self,
                                  source_field: np.ndarray,
                                  target_field: np.ndarray,
                                  test_points: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute interpolation error metrics.
        Args:
            source_field: Original field
            target_field: Interpolated field
            test_points: Points for error evaluation
        Returns:
            Error metrics
        """
        if test_points is None:
            # Use subset of source points
            n_test = min(100, len(self.source_nodes))
            indices = np.random.choice(len(self.source_nodes), n_test, replace=False)
            test_points = self.source_nodes[indices]
            test_values = source_field[indices]
        else:
            # Interpolate source field to test points
            test_values = self.interpolate_scalar_field(source_field)
        # Interpolate target field to test points
        temp_interpolator = FieldInterpolator(self.params)
        temp_interpolator.setup_interpolation(
            {'nodes': self.target_nodes},
            {'nodes': test_points}
        )
        interpolated_values = temp_interpolator.interpolate_scalar_field(target_field)
        # Compute errors
        absolute_error = np.abs(test_values - interpolated_values)
        relative_error = absolute_error / (np.abs(test_values) + 1e-12)
        return {
            'max_absolute_error': np.max(absolute_error),
            'mean_absolute_error': np.mean(absolute_error),
            'max_relative_error': np.max(relative_error),
            'mean_relative_error': np.mean(relative_error),
            'rms_error': np.sqrt(np.mean(absolute_error**2))
        }
class ConservationChecker:
    """Conservation law verification for multiphysics.
    Verifies conservation of mass, momentum, energy, and
    other physical quantities in coupled systems.
    """
    def __init__(self):
        """Initialize conservation checker."""
        self.tolerance = 1e-6
    def check_mass_conservation(self,
                              velocity_field: np.ndarray,
                              density_field: np.ndarray,
                              mesh: Dict[str, np.ndarray],
                              sources: Optional[np.ndarray] = None) -> ConservationResult:
        """Check mass conservation (continuity equation).
        Args:
            velocity_field: Velocity field
            density_field: Density field
            mesh: Computational mesh
            sources: Mass source terms
        Returns:
            Conservation check result
        """
        # Mass conservation: ∂ρ/∂t + ∇·(ρv) = S
        # Compute divergence of mass flux
        mass_flux = density_field[:, np.newaxis] * velocity_field
        divergence = self._compute_divergence(mass_flux, mesh)
        # Total mass flow out
        total_outflow = np.sum(divergence)
        # Account for sources
        if sources is not None:
            total_sources = np.sum(sources)
        else:
            total_sources = 0.0
        # Conservation error
        conservation_error = total_outflow - total_sources
        relative_error = abs(conservation_error) / (abs(total_sources) + 1e-12)
        conserved = relative_error < self.tolerance
        return ConservationResult(
            conserved=conserved,
            relative_error=relative_error,
            absolute_error=abs(conservation_error),
            conservation_type="mass",
            details={
                'total_outflow': total_outflow,
                'total_sources': total_sources,
                'divergence_field': divergence
            }
        )
    def check_energy_conservation(self,
                                velocity_field: np.ndarray,
                                temperature_field: np.ndarray,
                                heat_flux: np.ndarray,
                                mesh: Dict[str, np.ndarray],
                                heat_sources: Optional[np.ndarray] = None) -> ConservationResult:
        """Check energy conservation.
        Args:
            velocity_field: Velocity field
            temperature_field: Temperature field
            heat_flux: Heat flux vector
            mesh: Computational mesh
            heat_sources: Heat source terms
        Returns:
            Energy conservation result
        """
        # Energy conservation: ρcp ∂T/∂t + ∇·q = S
        # where q is heat flux
        # Divergence of heat flux
        heat_flux_div = self._compute_divergence(heat_flux, mesh)
        # Total heat outflow
        total_heat_outflow = np.sum(heat_flux_div)
        # Heat sources
        if heat_sources is not None:
            total_heat_sources = np.sum(heat_sources)
        else:
            total_heat_sources = 0.0
        # Conservation error
        energy_error = total_heat_outflow - total_heat_sources
        relative_error = abs(energy_error) / (abs(total_heat_sources) + 1e-12)
        conserved = relative_error < self.tolerance
        return ConservationResult(
            conserved=conserved,
            relative_error=relative_error,
            absolute_error=abs(energy_error),
            conservation_type="energy",
            details={
                'heat_outflow': total_heat_outflow,
                'heat_sources': total_heat_sources,
                'heat_flux_divergence': heat_flux_div
            }
        )
    def check_momentum_conservation(self,
                                  velocity_field: np.ndarray,
                                  pressure_field: np.ndarray,
                                  stress_tensor: np.ndarray,
                                  mesh: Dict[str, np.ndarray],
                                  body_forces: Optional[np.ndarray] = None) -> ConservationResult:
        """Check momentum conservation.
        Args:
            velocity_field: Velocity field
            pressure_field: Pressure field
            stress_tensor: Stress tensor field
            mesh: Computational mesh
            body_forces: Body force terms
        Returns:
            Momentum conservation result
        """
        # Momentum conservation: ρ ∂v/∂t + ∇·σ = F
        # where σ is stress tensor, F is body force
        # Compute stress divergence
        n_nodes = len(velocity_field)
        stress_div = np.zeros_like(velocity_field)
        for i in range(velocity_field.shape[1]):  # Each velocity component
            stress_column = stress_tensor[:, i, :]  # Extract stress column
            stress_div[:, i] = self._compute_divergence(stress_column, mesh)
        # Total momentum imbalance
        total_stress_div = np.sum(stress_div, axis=0)
        # Body forces
        if body_forces is not None:
            total_body_forces = np.sum(body_forces, axis=0)
        else:
            total_body_forces = np.zeros(velocity_field.shape[1])
        # Conservation error (per component)
        momentum_errors = total_stress_div - total_body_forces
        total_error = np.linalg.norm(momentum_errors)
        force_magnitude = np.linalg.norm(total_body_forces)
        relative_error = total_error / (force_magnitude + 1e-12)
        conserved = relative_error < self.tolerance
        return ConservationResult(
            conserved=conserved,
            relative_error=relative_error,
            absolute_error=total_error,
            conservation_type="momentum",
            details={
                'stress_divergence': total_stress_div,
                'body_forces': total_body_forces,
                'component_errors': momentum_errors
            }
        )
    def _compute_divergence(self,
                          vector_field: np.ndarray,
                          mesh: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute divergence of vector field.
        Args:
            vector_field: Vector field (n_nodes × n_dims)
            mesh: Computational mesh
        Returns:
            Divergence at each node
        """
        n_nodes = len(vector_field)
        divergence = np.zeros(n_nodes)
        # Simplified finite difference divergence
        # In practice, use proper finite element or finite volume computation
        nodes = mesh['nodes']
        h = 0.01  # Grid spacing estimate
        if vector_field.ndim == 1:
            # Scalar field - compute gradient magnitude
            for i in range(1, n_nodes - 1):
                divergence[i] = (vector_field[i+1] - vector_field[i-1]) / (2 * h)
        else:
            # Vector field - compute actual divergence
            for i in range(1, n_nodes - 1):
                for dim in range(vector_field.shape[1]):
                    if dim < nodes.shape[1]:  # Within mesh dimensions
                        grad_component = (vector_field[i+1, dim] - vector_field[i-1, dim]) / (2 * h)
                        divergence[i] += grad_component
        return divergence
class ConvergenceDiagnostics:
    """Convergence analysis tools for multiphysics.
    Provides tools for analyzing convergence behavior
    of coupled iterative solvers.
    """
    def __init__(self):
        """Initialize convergence diagnostics."""
        pass
    def analyze_convergence_history(self,
                                  residual_history: List[float],
                                  tolerance: float = 1e-6) -> Dict[str, Any]:
        """Analyze convergence history.
        Args:
            residual_history: History of residuals
            tolerance: Convergence tolerance
        Returns:
            Convergence analysis results
        """
        if len(residual_history) < 2:
            return {'status': 'insufficient_data'}
        residuals = np.array(residual_history)
        iterations = np.arange(len(residuals))
        # Convergence metrics
        converged = residuals[-1] < tolerance
        final_residual = residuals[-1]
        initial_residual = residuals[0]
        reduction_factor = final_residual / initial_residual if initial_residual > 0 else 0
        # Convergence rate analysis
        if len(residuals) > 2:
            # Fit exponential decay: r_k = r_0 * ρ^k
            log_residuals = np.log(residuals + 1e-16)  # Avoid log(0)
            # Linear fit to log(residuals)
            coeffs = np.polyfit(iterations, log_residuals, 1)
            convergence_rate = -coeffs[0]  # Negative slope
            # R-squared for fit quality
            log_residuals_fit = np.polyval(coeffs, iterations)
            ss_res = np.sum((log_residuals - log_residuals_fit)**2)
            ss_tot = np.sum((log_residuals - np.mean(log_residuals))**2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        else:
            convergence_rate = 0
            r_squared = 0
        # Detect stagnation
        if len(residuals) > 5:
            recent_change = abs(residuals[-1] - residuals[-5]) / residuals[-5]
            stagnated = recent_change < 0.01  # Less than 1% change
        else:
            stagnated = False
        # Oscillation detection
        if len(residuals) > 4:
            oscillating = self._detect_oscillation(residuals[-10:])
        else:
            oscillating = False
        return {
            'converged': converged,
            'final_residual': final_residual,
            'reduction_factor': reduction_factor,
            'convergence_rate': convergence_rate,
            'r_squared': r_squared,
            'stagnated': stagnated,
            'oscillating': oscillating,
            'iterations': len(residuals)
        }
    def _detect_oscillation(self, residuals: np.ndarray) -> bool:
        """Detect oscillatory behavior in residuals."""
        if len(residuals) < 4:
            return False
        # Count sign changes in residual differences
        diffs = np.diff(residuals)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        # High number of sign changes indicates oscillation
        return sign_changes > len(diffs) * 0.5
    def estimate_convergence_time(self,
                                residual_history: List[float],
                                tolerance: float,
                                max_iterations: int = 1000) -> Dict[str, Any]:
        """Estimate time to convergence.
        Args:
            residual_history: Current residual history
            tolerance: Target tolerance
            max_iterations: Maximum allowed iterations
        Returns:
            Convergence time estimate
        """
        analysis = self.analyze_convergence_history(residual_history, tolerance)
        if analysis.get('converged', False):
            return {
                'already_converged': True,
                'iterations_taken': len(residual_history)
            }
        convergence_rate = analysis.get('convergence_rate', 0)
        if convergence_rate <= 0:
            return {
                'will_converge': False,
                'reason': 'no_convergence_detected'
            }
        # Estimate iterations needed: r_0 * exp(-rate * k) = tolerance
        current_residual = residual_history[-1]
        if current_residual <= tolerance:
            return {'already_converged': True}
        estimated_iterations = np.log(tolerance / current_residual) / (-convergence_rate)
        total_iterations = len(residual_history) + estimated_iterations
        will_converge = total_iterations <= max_iterations
        return {
            'will_converge': will_converge,
            'estimated_additional_iterations': int(estimated_iterations),
            'estimated_total_iterations': int(total_iterations),
            'confidence': analysis.get('r_squared', 0)
        }
# Utility functions
def interpolate_field(source_mesh: Dict[str, np.ndarray],
                     target_mesh: Dict[str, np.ndarray],
                     field_data: np.ndarray,
                     method: str = "linear") -> np.ndarray:
    """Interpolate field between meshes.
    Args:
        source_mesh: Source mesh
        target_mesh: Target mesh
        field_data: Field values at source nodes
        method: Interpolation method
    Returns:
        Interpolated field at target nodes
    """
    params = InterpolationParams(method=method)
    interpolator = FieldInterpolator(params)
    interpolator.setup_interpolation(source_mesh, target_mesh)
    if field_data.ndim == 1:
        return interpolator.interpolate_scalar_field(field_data)
    else:
        return interpolator.interpolate_vector_field(field_data)
def project_field(source_field: np.ndarray,
                 source_space: str,
                 target_space: str,
                 mesh: Dict[str, np.ndarray]) -> np.ndarray:
    """Project field between function spaces.
    Args:
        source_field: Field in source space
        source_space: Source function space
        target_space: Target function space
        mesh: Computational mesh
    Returns:
        Projected field in target space
    """
    # Simplified projection - in practice, use proper FE projection
    if source_space == target_space:
        return source_field.copy()
    if source_space == "nodal" and target_space == "elemental":
        # Node to element projection (averaging)
        if 'elements' in mesh:
            elements = mesh['elements']
            n_elements = len(elements)
            projected = np.zeros(n_elements)
            for i, element in enumerate(elements):
                projected[i] = np.mean(source_field[element])
        else:
            # Fallback: simple averaging of adjacent nodes
            n_target = len(source_field) // 2
            projected = np.zeros(n_target)
            for i in range(n_target):
                projected[i] = np.mean(source_field[2*i:2*i+2])
    elif source_space == "elemental" and target_space == "nodal":
        # Element to node projection (distribution)
        n_nodes = len(mesh['nodes'])
        projected = np.zeros(n_nodes)
        if 'elements' in mesh:
            elements = mesh['elements']
            for i, element in enumerate(elements):
                for node in element:
                    projected[node] += source_field[i]
            # Average contributions
            node_count = np.zeros(n_nodes)
            for element in elements:
                for node in element:
                    node_count[node] += 1
            projected = projected / (node_count + 1e-12)
        else:
            # Fallback: duplicate elements to nodes
            projected = np.repeat(source_field, 2)[:n_nodes]
    else:
        raise ValueError(f"Unsupported projection: {source_space} -> {target_space}")
    return projected
def check_conservation(solution_data: Dict[str, np.ndarray],
                      mesh: Dict[str, np.ndarray],
                      conservation_type: str = "mass") -> ConservationResult:
    """Check conservation law.
    Args:
        solution_data: Solution fields
        mesh: Computational mesh
        conservation_type: Type of conservation law
    Returns:
        Conservation check result
    """
    checker = ConservationChecker()
    if conservation_type == "mass":
        velocity = solution_data.get('velocity', np.zeros((len(mesh['nodes']), 2)))
        density = solution_data.get('density', np.ones(len(mesh['nodes'])))
        sources = solution_data.get('mass_sources', None)
        return checker.check_mass_conservation(velocity, density, mesh, sources)
    elif conservation_type == "energy":
        velocity = solution_data.get('velocity', np.zeros((len(mesh['nodes']), 2)))
        temperature = solution_data.get('temperature', np.ones(len(mesh['nodes'])) * 300)
        heat_flux = solution_data.get('heat_flux', np.zeros((len(mesh['nodes']), 2)))
        heat_sources = solution_data.get('heat_sources', None)
        return checker.check_energy_conservation(velocity, temperature, heat_flux, mesh, heat_sources)
    elif conservation_type == "momentum":
        velocity = solution_data.get('velocity', np.zeros((len(mesh['nodes']), 2)))
        pressure = solution_data.get('pressure', np.zeros(len(mesh['nodes'])))
        stress = solution_data.get('stress', np.zeros((len(mesh['nodes']), 2, 2)))
        body_forces = solution_data.get('body_forces', None)
        return checker.check_momentum_conservation(velocity, pressure, stress, mesh, body_forces)
    else:
        raise ValueError(f"Unknown conservation type: {conservation_type}")
def compute_interface_normal(interface_nodes: np.ndarray,
                           mesh: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute normal vectors at interface.
    Args:
        interface_nodes: Interface node coordinates
        mesh: Full mesh data
    Returns:
        Normal vectors at interface nodes
    """
    n_interface = len(interface_nodes)
    normals = np.zeros((n_interface, interface_nodes.shape[1]))
    if interface_nodes.shape[1] == 2:  # 2D
        for i in range(n_interface):
            if i == 0:
                # First node
                tangent = interface_nodes[1] - interface_nodes[0]
            elif i == n_interface - 1:
                # Last node
                tangent = interface_nodes[-1] - interface_nodes[-2]
            else:
                # Interior node
                tangent = interface_nodes[i+1] - interface_nodes[i-1]
            # Normal is perpendicular to tangent
            normal = np.array([-tangent[1], tangent[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-12)
            normals[i] = normal
    elif interface_nodes.shape[1] == 3:  # 3D
        # For 3D, need surface triangulation to compute normals
        warnings.warn("3D interface normal computation not fully implemented")
        normals = np.zeros((n_interface, 3))
        normals[:, 2] = 1.0  # Default to z-direction
    return normals
def transfer_data(source_interface: CouplingInterface,
                 source_data: np.ndarray,
                 target_mesh: Dict[str, np.ndarray],
                 method: str = "conservative") -> np.ndarray:
    """Transfer data between domains through interface.
    Args:
        source_interface: Source coupling interface
        source_data: Data at source interface
        target_mesh: Target mesh
        method: Transfer method
    Returns:
        Transferred data at target
    """
    if method == "conservative":
        # Conservative transfer preserves integral quantities
        return source_interface.transfer_field(source_data)
    elif method == "consistent":
        # Consistent transfer preserves point values
        # Use interpolation
        source_mesh = {'nodes': source_interface.source_nodes}
        target_mesh_interface = {'nodes': source_interface.target_nodes}
        return interpolate_field(source_mesh, target_mesh_interface, source_data)
    else:
        raise ValueError(f"Unknown transfer method: {method}")
def compute_coupling_residual(solution_a: np.ndarray,
                            solution_b: np.ndarray,
                            interface: CouplingInterface) -> float:
    """Compute coupling residual at interface.
    Args:
        solution_a: Solution from domain A
        solution_b: Solution from domain B
        interface: Coupling interface
    Returns:
        Coupling residual magnitude
    """
    # Extract interface values
    interface_a = interface.extract_interface_data(solution_a, "source")
    interface_b = interface.extract_interface_data(solution_b, "target")
    # Compute residual
    residual = interface_a - interface_b
    return np.linalg.norm(residual)