"""Fluid-Structure Interaction (FSI) Module.
This module implements fluid-structure interaction methods including
Arbitrary Lagrangian-Eulerian (ALE) formulations, mesh motion algorithms,
and FSI benchmarks.
Classes:
    FluidStructureInteraction: Main FSI solver
    FluidSolver: Fluid domain solver
    StructuralSolver: Structural domain solver
    MeshMotion: Mesh motion algorithms
    ALE: Arbitrary Lagrangian-Eulerian framework
Functions:
    fsi_benchmark: Standard FSI benchmark problems
    vortex_induced_vibration: VIV analysis
Author: Berkeley SciComp Team
Date: 2024
"""
import numpy as np
from typing import Dict, Tuple, Optional, Callable, Any, List
from dataclasses import dataclass
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
import warnings
from .coupling import CoupledSystem, CouplingInterface, CouplingScheme
@dataclass
class FluidProperties:
    """Fluid material properties."""
    density: float  # kg/m³
    viscosity: float  # Pa·s
    bulk_modulus: Optional[float] = None  # Pa (for compressible)
@dataclass
class StructuralProperties:
    """Structural material properties."""
    density: float  # kg/m³
    youngs_modulus: float  # Pa
    poissons_ratio: float
    damping_ratio: float = 0.0
    @property
    def shear_modulus(self) -> float:
        """Compute shear modulus."""
        return self.youngs_modulus / (2 * (1 + self.poissons_ratio))
    @property
    def bulk_modulus(self) -> float:
        """Compute bulk modulus."""
        return self.youngs_modulus / (3 * (1 - 2 * self.poissons_ratio))
class FluidSolver:
    """Incompressible fluid flow solver.
    Implements finite element solution of Navier-Stokes equations
    with support for moving meshes (ALE formulation).
    """
    def __init__(self,
                 mesh: Dict[str, np.ndarray],
                 properties: FluidProperties,
                 formulation: str = "incompressible"):
        """Initialize fluid solver.
        Args:
            mesh: Mesh dictionary with 'nodes', 'elements'
            properties: Fluid properties
            formulation: Flow formulation type
        """
        self.mesh = mesh
        self.properties = properties
        self.formulation = formulation
        # Solution variables
        self.velocity = None
        self.pressure = None
        self.mesh_velocity = None
        # System matrices
        self.mass_matrix = None
        self.stiffness_matrix = None
        self.gradient_matrix = None
        self.divergence_matrix = None
        # Time integration
        self.dt = 0.01
        self.theta = 0.5  # Crank-Nicolson by default
        self._setup_system()
    def _setup_system(self):
        """Setup finite element system."""
        n_nodes = len(self.mesh['nodes'])
        n_elements = len(self.mesh['elements'])
        # Initialize solution
        self.velocity = np.zeros((n_nodes, 2))  # 2D for now
        self.pressure = np.zeros(n_nodes)
        # Build system matrices (placeholder)
        # In practice, use proper FE assembly
        self.mass_matrix = sparse.eye(n_nodes * 2) * self.properties.density
        self.stiffness_matrix = sparse.eye(n_nodes * 2) * self.properties.viscosity
    def solve(self, dt: float, boundary_conditions: Dict[str, Any],
              interface_data: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
        """Solve fluid flow for one time step.
        Args:
            dt: Time step size
            boundary_conditions: Boundary condition data
            interface_data: FSI interface data
        Returns:
            Dictionary with velocity and pressure fields
        """
        self.dt = dt
        # Update mesh if ALE
        if self.mesh_velocity is not None:
            self._update_mesh(dt)
        # Apply interface conditions
        if interface_data:
            self._apply_interface_conditions(interface_data)
        # Solve Navier-Stokes
        if self.formulation == "incompressible":
            self._solve_incompressible_ns(boundary_conditions)
        else:
            raise NotImplementedError(f"Formulation {self.formulation} not implemented")
        return {
            'velocity': self.velocity.copy(),
            'pressure': self.pressure.copy()
        }
    def _solve_incompressible_ns(self, bc: Dict[str, Any]):
        """Solve incompressible Navier-Stokes equations."""
        # Simplified implementation
        # In practice, use projection method or coupled solver
        # Momentum equation
        # M * (v^{n+1} - v^n)/dt + K * v^{n+1} + G * p^{n+1} = f
        # Continuity equation
        # D * v^{n+1} = 0
        # For demonstration, use simple explicit update
        n_nodes = len(self.mesh['nodes'])
        # Advection term (simplified)
        advection = np.zeros_like(self.velocity)
        for i in range(2):  # 2D
            advection[:, i] = self.velocity[:, 0] * np.gradient(self.velocity[:, i])[0]
        # Viscous term (simplified Laplacian)
        viscous = np.zeros_like(self.velocity)
        # In practice, compute proper Laplacian
        # Pressure gradient (simplified)
        pressure_grad = np.zeros_like(self.velocity)
        pressure_grad[:, 0] = np.gradient(self.pressure)[0]
        # Update velocity
        self.velocity += dt * (-advection +
                               self.properties.viscosity / self.properties.density * viscous -
                               1.0 / self.properties.density * pressure_grad)
        # Apply boundary conditions
        self._apply_boundary_conditions(bc)
        # Pressure correction (simplified)
        # In practice, solve Poisson equation
        divergence = np.gradient(self.velocity[:, 0])[0]
        self.pressure += dt * self.properties.bulk_modulus * divergence if self.properties.bulk_modulus else 0
    def _apply_interface_conditions(self, interface_data: Dict[str, Any]):
        """Apply FSI interface conditions."""
        for name, data in interface_data.items():
            if 'interface' in data and 'displacement' in data:
                interface = data['interface']
                displacement = data['displacement']
                # Set mesh velocity at interface
                if self.mesh_velocity is None:
                    self.mesh_velocity = np.zeros_like(self.velocity)
                # Map structural displacement to fluid mesh velocity
                # This is simplified - use proper mapping
                interface_nodes = data.get('nodes', [])
                if interface_nodes:
                    self.mesh_velocity[interface_nodes] = displacement / self.dt
    def _apply_boundary_conditions(self, bc: Dict[str, Any]):
        """Apply boundary conditions."""
        # Inlet
        if 'inlet' in bc:
            inlet_nodes = bc['inlet']['nodes']
            inlet_velocity = bc['inlet']['velocity']
            self.velocity[inlet_nodes] = inlet_velocity
        # Wall (no-slip)
        if 'wall' in bc:
            wall_nodes = bc['wall']['nodes']
            self.velocity[wall_nodes] = 0
        # Outlet
        if 'outlet' in bc:
            # Natural boundary condition for pressure
            pass
    def _update_mesh(self, dt: float):
        """Update mesh position for ALE."""
        if self.mesh_velocity is not None:
            self.mesh['nodes'] += dt * self.mesh_velocity[:, :2]  # Update only spatial coords
    def compute_interface_forces(self, interface_nodes: List[int]) -> np.ndarray:
        """Compute forces on fluid-structure interface.
        Args:
            interface_nodes: List of interface node indices
        Returns:
            Force vector at interface nodes
        """
        # Compute stress tensor
        # σ = -pI + μ(∇v + ∇v^T)
        forces = np.zeros((len(interface_nodes), 2))
        for i, node in enumerate(interface_nodes):
            # Pressure contribution
            forces[i] = -self.pressure[node] * np.array([1, 0])  # Simplified normal
            # Viscous contribution
            # In practice, compute proper velocity gradients
        return forces
class StructuralSolver:
    """Structural dynamics solver.
    Implements finite element solution for structural dynamics
    with support for large deformations and FSI coupling.
    """
    def __init__(self,
                 mesh: Dict[str, np.ndarray],
                 properties: StructuralProperties,
                 formulation: str = "linear"):
        """Initialize structural solver.
        Args:
            mesh: Mesh dictionary
            properties: Structural properties
            formulation: Formulation type (linear/nonlinear)
        """
        self.mesh = mesh
        self.properties = properties
        self.formulation = formulation
        # Solution variables
        self.displacement = None
        self.velocity = None
        self.acceleration = None
        # System matrices
        self.mass_matrix = None
        self.stiffness_matrix = None
        self.damping_matrix = None
        # Time integration (Newmark)
        self.beta = 0.25
        self.gamma = 0.5
        self._setup_system()
    def _setup_system(self):
        """Setup finite element system."""
        n_nodes = len(self.mesh['nodes'])
        n_dof = n_nodes * 2  # 2D
        # Initialize solution
        self.displacement = np.zeros((n_nodes, 2))
        self.velocity = np.zeros((n_nodes, 2))
        self.acceleration = np.zeros((n_nodes, 2))
        # Build system matrices (simplified)
        self.mass_matrix = sparse.eye(n_dof) * self.properties.density
        self.stiffness_matrix = sparse.eye(n_dof) * self.properties.youngs_modulus
        # Rayleigh damping
        alpha = 0.01  # Mass proportional
        beta = 0.01   # Stiffness proportional
        self.damping_matrix = alpha * self.mass_matrix + beta * self.stiffness_matrix
    def solve(self, dt: float, external_forces: np.ndarray,
              interface_data: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
        """Solve structural dynamics for one time step.
        Args:
            dt: Time step size
            external_forces: External force vector
            interface_data: FSI interface data
        Returns:
            Dictionary with displacement, velocity, acceleration
        """
        # Apply interface forces
        total_forces = external_forces.copy()
        if interface_data:
            for name, data in interface_data.items():
                if 'forces' in data:
                    interface_forces = data['forces']
                    interface_nodes = data.get('nodes', [])
                    # Add interface forces
                    for i, node in enumerate(interface_nodes):
                        total_forces[node] += interface_forces[i]
        # Newmark time integration
        if self.formulation == "linear":
            self._newmark_linear(dt, total_forces)
        else:
            self._newmark_nonlinear(dt, total_forces)
        return {
            'displacement': self.displacement.copy(),
            'velocity': self.velocity.copy(),
            'acceleration': self.acceleration.copy()
        }
    def _newmark_linear(self, dt: float, forces: np.ndarray):
        """Linear Newmark time integration."""
        # Predict
        disp_pred = self.displacement + dt * self.velocity + \
                   dt**2 * (0.5 - self.beta) * self.acceleration
        vel_pred = self.velocity + dt * (1 - self.gamma) * self.acceleration
        # Flatten for matrix operations
        n_dof = self.displacement.size
        disp_vec = self.displacement.flatten()
        force_vec = forces.flatten()
        # Effective stiffness
        K_eff = self.stiffness_matrix + \
                self.damping_matrix * (self.gamma / (self.beta * dt)) + \
                self.mass_matrix * (1.0 / (self.beta * dt**2))
        # Effective force
        F_eff = force_vec
        # Add contributions from prediction (simplified)
        # Solve
        # K_eff * u = F_eff
        # In practice, use proper linear solver
        disp_new = sparse_linalg.spsolve(K_eff, F_eff)
        # Update acceleration and velocity
        self.acceleration = (disp_new - disp_vec) / (self.beta * dt**2) - \
                          self.velocity.flatten() / (self.beta * dt) - \
                          self.acceleration.flatten() * (1 - 1/(2*self.beta))
        self.velocity = vel_pred.flatten() + \
                       dt * self.gamma * self.acceleration
        # Reshape
        self.displacement = disp_new.reshape(self.displacement.shape)
        self.velocity = self.velocity.reshape(self.displacement.shape)
        self.acceleration = self.acceleration.reshape(self.displacement.shape)
    def _newmark_nonlinear(self, dt: float, forces: np.ndarray):
        """Nonlinear Newmark time integration."""
        # Implement Newton-Raphson iteration
        warnings.warn("Nonlinear structural dynamics not fully implemented")
        self._newmark_linear(dt, forces)  # Fallback to linear
class MeshMotion:
    """Mesh motion algorithms for ALE formulation.
    Provides various mesh motion strategies including
    Laplacian smoothing, elastic analogy, and RBF interpolation.
    """
    def __init__(self, method: str = "laplacian"):
        """Initialize mesh motion solver.
        Args:
            method: Mesh motion method
        """
        self.method = method
        self.reference_mesh = None
    def set_reference_mesh(self, mesh: Dict[str, np.ndarray]):
        """Set reference mesh configuration."""
        self.reference_mesh = {
            'nodes': mesh['nodes'].copy(),
            'elements': mesh['elements'].copy()
        }
    def compute_mesh_displacement(self,
                                 boundary_displacement: Dict[int, np.ndarray],
                                 mesh: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute mesh displacement field.
        Args:
            boundary_displacement: Prescribed boundary displacements
            mesh: Current mesh
        Returns:
            Displacement field for all nodes
        """
        n_nodes = len(mesh['nodes'])
        displacement = np.zeros((n_nodes, 2))  # 2D
        # Set boundary displacements
        for node_id, disp in boundary_displacement.items():
            displacement[node_id] = disp
        if self.method == "laplacian":
            # Solve Laplace equation for mesh motion
            # ∇²u = 0 with prescribed boundary conditions
            # Build Laplacian matrix (simplified)
            # In practice, use proper FE assembly
            interior_nodes = [i for i in range(n_nodes)
                            if i not in boundary_displacement]
            n_interior = len(interior_nodes)
            L = sparse.diags([-1, 4, -1], [-1, 0, 1],
                           shape=(n_interior, n_interior))
            # Solve for each component
            for dim in range(2):
                rhs = np.zeros(n_interior)
                # Add boundary contributions to RHS
                # Solve
                interior_disp = sparse_linalg.spsolve(L, rhs)
                for i, node in enumerate(interior_nodes):
                    displacement[node, dim] = interior_disp[i]
        elif self.method == "rbf":
            # Radial basis function interpolation
            from scipy.interpolate import RBFInterpolator
            # Known displacements
            known_nodes = list(boundary_displacement.keys())
            known_disp = np.array([boundary_displacement[i] for i in known_nodes])
            known_pos = mesh['nodes'][known_nodes]
            # Interpolate
            rbf = RBFInterpolator(known_pos, known_disp)
            displacement = rbf(mesh['nodes'])
        return displacement
    def update_mesh_quality(self, mesh: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Compute mesh quality metrics.
        Args:
            mesh: Current mesh
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        # Compute element qualities
        elements = mesh['elements']
        nodes = mesh['nodes']
        qualities = []
        for elem in elements:
            # Compute element quality (e.g., aspect ratio)
            vertices = nodes[elem]
            # Simple quality metric for triangles
            if len(elem) == 3:
                # Area
                area = 0.5 * abs(np.cross(vertices[1] - vertices[0],
                                         vertices[2] - vertices[0]))
                # Perimeter
                perimeter = (np.linalg.norm(vertices[1] - vertices[0]) +
                           np.linalg.norm(vertices[2] - vertices[1]) +
                           np.linalg.norm(vertices[0] - vertices[2]))
                # Quality (normalized)
                quality = 4 * np.sqrt(3) * area / (perimeter**2)
                qualities.append(quality)
        metrics['min_quality'] = np.min(qualities)
        metrics['mean_quality'] = np.mean(qualities)
        metrics['elements_below_threshold'] = np.sum(np.array(qualities) < 0.1)
        return metrics
class ALE:
    """Arbitrary Lagrangian-Eulerian framework.
    Handles the ALE formulation for moving mesh problems
    including convective terms and geometric conservation law.
    """
    def __init__(self):
        """Initialize ALE framework."""
        self.mesh_velocity = None
        self.reference_configuration = None
    def compute_mesh_velocity(self,
                            mesh_displacement: np.ndarray,
                            dt: float) -> np.ndarray:
        """Compute mesh velocity from displacement.
        Args:
            mesh_displacement: Mesh displacement field
            dt: Time step
        Returns:
            Mesh velocity field
        """
        if self.mesh_velocity is None:
            # First time step
            self.mesh_velocity = mesh_displacement / dt
        else:
            # Use previous velocity for smoothing
            alpha = 0.5  # Smoothing parameter
            new_velocity = mesh_displacement / dt
            self.mesh_velocity = alpha * new_velocity + (1 - alpha) * self.mesh_velocity
        return self.mesh_velocity
    def compute_convective_velocity(self,
                                  material_velocity: np.ndarray,
                                  mesh_velocity: np.ndarray) -> np.ndarray:
        """Compute convective velocity for ALE formulation.
        Args:
            material_velocity: Material point velocity
            mesh_velocity: Mesh velocity
        Returns:
            Convective velocity (u - u_mesh)
        """
        return material_velocity - mesh_velocity
    def check_geometric_conservation(self,
                                   mesh_old: Dict[str, np.ndarray],
                                   mesh_new: Dict[str, np.ndarray],
                                   dt: float) -> float:
        """Check geometric conservation law.
        Args:
            mesh_old: Previous mesh configuration
            mesh_new: Current mesh configuration
            dt: Time step
        Returns:
            GCL error measure
        """
        # Compute volume change
        volume_old = self._compute_mesh_volume(mesh_old)
        volume_new = self._compute_mesh_volume(mesh_new)
        # GCL: dV/dt = ∫ u_mesh · n dS
        volume_rate = (volume_new - volume_old) / dt
        # Compute surface integral of mesh velocity (simplified)
        # In practice, compute proper surface integral
        gcl_error = abs(volume_rate)  # Simplified
        return gcl_error
    def _compute_mesh_volume(self, mesh: Dict[str, np.ndarray]) -> float:
        """Compute total mesh volume/area."""
        total_volume = 0.0
        for elem in mesh['elements']:
            vertices = mesh['nodes'][elem]
            if len(elem) == 3:  # Triangle
                # Area
                area = 0.5 * abs(np.cross(vertices[1] - vertices[0],
                                        vertices[2] - vertices[0]))
                total_volume += area
        return total_volume
class FluidStructureInteraction(CoupledSystem):
    """Main FSI solver class.
    Coordinates fluid and structural solvers with
    appropriate coupling strategies.
    """
    def __init__(self,
                 fluid_mesh: Dict[str, np.ndarray],
                 structure_mesh: Dict[str, np.ndarray],
                 fluid_props: FluidProperties,
                 structure_props: StructuralProperties,
                 coupling_scheme: CouplingScheme = CouplingScheme.PARTITIONED_IMPLICIT):
        """Initialize FSI solver.
        Args:
            fluid_mesh: Fluid domain mesh
            structure_mesh: Structure domain mesh
            fluid_props: Fluid properties
            structure_props: Structural properties
            coupling_scheme: FSI coupling scheme
        """
        super().__init__("FSI", ["fluid", "structure"], coupling_scheme)
        self.fluid_mesh = fluid_mesh
        self.structure_mesh = structure_mesh
        self.fluid_props = fluid_props
        self.structure_props = structure_props
        # Additional FSI components
        self.mesh_motion = MeshMotion("rbf")
        self.ale = ALE()
        self.setup_domains()
        self.setup_coupling()
    def setup_domains(self):
        """Setup fluid and structural domains."""
        # Create fluid solver
        self.fluid_solver = FluidSolver(
            self.fluid_mesh,
            self.fluid_props,
            formulation="incompressible"
        )
        self.add_domain_solver("fluid", self.fluid_solver)
        # Create structural solver
        self.structure_solver = StructuralSolver(
            self.structure_mesh,
            self.structure_props,
            formulation="linear"
        )
        self.add_domain_solver("structure", self.structure_solver)
    def setup_coupling(self):
        """Setup FSI coupling interface."""
        # Define interface (simplified)
        # In practice, identify interface nodes properly
        # For demonstration, assume interface nodes are given
        fluid_interface_nodes = []  # To be populated
        structure_interface_nodes = []  # To be populated
        # Create coupling interface
        interface = CouplingInterface(
            "fsi_interface",
            "structure",
            "fluid",
            "surface"
        )
        self.add_interface(interface)
    def solve_fsi_step(self, dt: float,
                      boundary_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Solve one FSI time step.
        Args:
            dt: Time step size
            boundary_conditions: BC for both domains
        Returns:
            FSI solution data
        """
        fluid_bc = boundary_conditions.get('fluid', {})
        structure_bc = boundary_conditions.get('structure', {})
        if self.coupling_scheme == CouplingScheme.PARTITIONED_IMPLICIT:
            # Implicit coupling with subiterations
            converged = False
            iteration = 0
            # Initialize interface quantities
            interface_displacement = np.zeros((10, 2))  # Placeholder size
            interface_forces = np.zeros((10, 2))
            while not converged and iteration < self.max_iterations:
                iteration += 1
                # Store previous values
                prev_displacement = interface_displacement.copy()
                # Solve structure with interface forces
                structure_result = self.structure_solver.solve(
                    dt, interface_forces
                )
                # Extract interface displacement
                interface_displacement = structure_result['displacement'][:10]  # Placeholder
                # Update fluid mesh
                mesh_displacement = self.mesh_motion.compute_mesh_displacement(
                    {i: interface_displacement[i] for i in range(10)},
                    self.fluid_mesh
                )
                # Solve fluid with moving mesh
                self.fluid_solver.mesh_velocity = self.ale.compute_mesh_velocity(
                    mesh_displacement, dt
                )
                fluid_result = self.fluid_solver.solve(
                    dt, fluid_bc, {'interface': interface_displacement}
                )
                # Compute interface forces from fluid
                interface_forces = self.fluid_solver.compute_interface_forces(
                    list(range(10))  # Placeholder interface nodes
                )
                # Check convergence
                displacement_change = np.linalg.norm(
                    interface_displacement - prev_displacement
                )
                if displacement_change < self.tolerance:
                    converged = True
            if not converged:
                warnings.warn(f"FSI did not converge after {iteration} iterations")
            return {
                'fluid': fluid_result,
                'structure': structure_result,
                'iterations': iteration,
                'converged': converged
            }
        else:
            raise NotImplementedError(f"Coupling scheme {self.coupling_scheme} not implemented for FSI")
# FSI Benchmark functions
def fsi_benchmark(benchmark_name: str = "turek_hron") -> Dict[str, Any]:
    """Standard FSI benchmark problems.
    Args:
        benchmark_name: Name of benchmark problem
    Returns:
        Benchmark configuration and reference data
    """
    if benchmark_name == "turek_hron":
        # Turek & Hron FSI benchmark
        config = {
            'geometry': {
                'channel_length': 2.5,
                'channel_height': 0.41,
                'cylinder_center': (0.2, 0.2),
                'cylinder_radius': 0.05,
                'beam_length': 0.35,
                'beam_height': 0.02
            },
            'fluid': {
                'density': 1000.0,  # kg/m³
                'viscosity': 1.0,   # Pa·s
                'inlet_velocity': lambda y: 1.5 * y * (0.41 - y) / (0.41/2)**2
            },
            'structure': {
                'density': 1000.0,  # kg/m³
                'youngs_modulus': 1.4e6,  # Pa
                'poissons_ratio': 0.4
            },
            'reference_results': {
                'displacement_amplitude': 0.0227,
                'frequency': 2.0
            }
        }
    elif benchmark_name == "lid_driven_cavity":
        # Lid-driven cavity with flexible bottom
        config = {
            'geometry': {
                'cavity_size': 1.0,
                'plate_thickness': 0.01
            },
            'fluid': {
                'density': 1.0,
                'viscosity': 0.01,
                'lid_velocity': 1.0
            },
            'structure': {
                'density': 100.0,
                'youngs_modulus': 1.0e3,
                'poissons_ratio': 0.3
            }
        }
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
    return config
def vortex_induced_vibration(reduced_velocity: float,
                           mass_ratio: float,
                           damping_ratio: float = 0.0) -> Dict[str, float]:
    """Analyze vortex-induced vibration.
    Args:
        reduced_velocity: U/(f_n*D) where U is flow velocity
        mass_ratio: m/(ρ*D²) where m is mass per unit length
        damping_ratio: Structural damping ratio
    Returns:
        VIV response characteristics
    """
    # Simplified VIV model
    # Lock-in range approximately 4 < U_r < 8
    if 4 < reduced_velocity < 8:
        # In lock-in range
        amplitude_ratio = 0.5 * np.exp(-damping_ratio * mass_ratio)
        frequency_ratio = 1.0  # Locked to natural frequency
    else:
        # Outside lock-in
        amplitude_ratio = 0.1 * np.exp(-damping_ratio * mass_ratio)
        frequency_ratio = 0.2 * reduced_velocity  # Strouhal relation
    return {
        'amplitude_ratio': amplitude_ratio,
        'frequency_ratio': frequency_ratio,
        'in_lock_in': 4 < reduced_velocity < 8
    }