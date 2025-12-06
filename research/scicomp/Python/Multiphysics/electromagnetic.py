"""Electromagnetic-Thermal Coupling Module.
This module implements electromagnetic-thermal coupling for problems involving
electromagnetic fields and heat generation, including Joule heating,
induction heating, and eddy current effects.
Classes:
    ElectromagneticThermalCoupling: Main coupled solver
    MaxwellSolver: Electromagnetic field solver
    JouleHeating: Joule heating model
    InductionHeating: Induction heating solver
    EddyCurrentSolver: Eddy current analysis
Functions:
    electromagnetic_heating: Compute EM heating
    coupled_em_thermal: Solve coupled EM-thermal problem
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
# Physical constants
MU_0 = 4 * np.pi * 1e-7  # Vacuum permeability (H/m)
EPSILON_0 = 8.854e-12    # Vacuum permittivity (F/m)
SIGMA_0 = 5.67e-8        # Stefan-Boltzmann constant (W/m²K⁴)
@dataclass
class ElectromagneticProperties:
    """Electromagnetic material properties."""
    conductivity: float  # S/m (electrical)
    permittivity: float = EPSILON_0  # F/m
    permeability: float = MU_0  # H/m
    @property
    def skin_depth(self, frequency: float) -> float:
        """Electromagnetic skin depth."""
        return 1.0 / np.sqrt(np.pi * frequency * self.permeability * self.conductivity)
    @property
    def impedance(self) -> float:
        """Wave impedance."""
        return np.sqrt(self.permeability / self.permittivity)
class MaxwellSolver:
    """Electromagnetic field solver.
    Solves Maxwell's equations for electromagnetic fields
    with support for various formulations.
    """
    def __init__(self,
                 mesh: Dict[str, np.ndarray],
                 properties: ElectromagneticProperties,
                 formulation: str = "A-phi"):
        """Initialize Maxwell solver.
        Args:
            mesh: Finite element mesh
            properties: EM properties
            formulation: Solution formulation (A-phi, E-B, etc.)
        """
        self.mesh = mesh
        self.properties = properties
        self.formulation = formulation
        # Solution variables
        self.vector_potential = None  # A
        self.scalar_potential = None  # φ
        self.electric_field = None    # E
        self.magnetic_field = None    # B
        self.current_density = None   # J
        # System matrices
        self.curl_curl_matrix = None
        self.mass_matrix = None
        self.conductivity_matrix = None
        # Frequency domain
        self.frequency = 0.0  # Hz (0 for DC)
        self.omega = 0.0      # Angular frequency
        self._setup_system()
    def _setup_system(self):
        """Setup finite element system for EM."""
        n_nodes = len(self.mesh['nodes'])
        n_edges = self._estimate_edges()  # Edge elements for vector fields
        if self.formulation == "A-phi":
            # Vector potential formulation
            self.vector_potential = np.zeros((n_nodes, 3))  # 3D vector
            self.scalar_potential = np.zeros(n_nodes)
            # Build matrices (simplified)
            self.curl_curl_matrix = sparse.eye(n_nodes * 3) / self.properties.permeability
            self.conductivity_matrix = sparse.eye(n_nodes * 3) * self.properties.conductivity
        elif self.formulation == "E":
            # Electric field formulation
            self.electric_field = np.zeros((n_edges, 3))
        else:
            raise NotImplementedError(f"Formulation {self.formulation} not implemented")
    def _estimate_edges(self) -> int:
        """Estimate number of edges for edge elements."""
        # Rough estimate: edges ≈ 3 * nodes for 3D
        return 3 * len(self.mesh['nodes'])
    def solve_static(self,
                    boundary_conditions: Dict[str, Any],
                    current_source: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Solve static (DC) electromagnetic problem.
        Args:
            boundary_conditions: EM boundary conditions
            current_source: Current density source
        Returns:
            EM field solution
        """
        n_nodes = len(self.mesh['nodes'])
        if self.formulation == "A-phi":
            # Magnetostatic: ∇×(1/μ ∇×A) = J
            # Simplified to Poisson equation for each component
            rhs = np.zeros(n_nodes * 3)
            if current_source is not None:
                rhs = current_source.flatten()
            # Apply boundary conditions
            self._apply_em_bc(boundary_conditions, self.curl_curl_matrix, rhs)
            # Solve
            a_flat = sparse_linalg.spsolve(self.curl_curl_matrix, rhs)
            self.vector_potential = a_flat.reshape((n_nodes, 3))
            # Compute magnetic field B = ∇×A
            self.magnetic_field = self._compute_curl(self.vector_potential)
            # For conductors, also solve for scalar potential
            # -∇·(σ∇φ) = 0 with J = -σ∇φ
            if self.properties.conductivity > 0:
                laplacian = sparse.eye(n_nodes) * self.properties.conductivity
                phi_rhs = np.zeros(n_nodes)
                # Apply potential BC
                if 'potential' in boundary_conditions:
                    self._apply_potential_bc(boundary_conditions['potential'],
                                           laplacian, phi_rhs)
                self.scalar_potential = sparse_linalg.spsolve(laplacian, phi_rhs)
                # Electric field
                self.electric_field = -self._compute_gradient(self.scalar_potential)
                # Current density
                self.current_density = self.properties.conductivity * self.electric_field
        return {
            'vector_potential': self.vector_potential,
            'scalar_potential': self.scalar_potential,
            'electric_field': self.electric_field,
            'magnetic_field': self.magnetic_field,
            'current_density': self.current_density
        }
    def solve_frequency_domain(self,
                             frequency: float,
                             boundary_conditions: Dict[str, Any],
                             current_source: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Solve frequency domain electromagnetic problem.
        Args:
            frequency: Frequency (Hz)
            boundary_conditions: EM boundary conditions
            current_source: Current source (complex)
        Returns:
            Complex EM field solution
        """
        self.frequency = frequency
        self.omega = 2 * np.pi * frequency
        n_nodes = len(self.mesh['nodes'])
        if self.formulation == "A-phi":
            # Time-harmonic: ∇×(1/μ ∇×A) + jωσA = J
            # where j is imaginary unit
            # System matrix (complex)
            system_matrix = self.curl_curl_matrix + \
                          1j * self.omega * self.conductivity_matrix
            rhs = np.zeros(n_nodes * 3, dtype=complex)
            if current_source is not None:
                rhs = current_source.flatten()
            # Apply BC
            self._apply_em_bc(boundary_conditions, system_matrix, rhs)
            # Solve complex system
            a_flat = sparse_linalg.spsolve(system_matrix, rhs)
            self.vector_potential = a_flat.reshape((n_nodes, 3))
            # Magnetic field
            self.magnetic_field = self._compute_curl(self.vector_potential)
            # Electric field from E = -jωA - ∇φ
            self.electric_field = -1j * self.omega * self.vector_potential
            if self.scalar_potential is not None:
                self.electric_field -= self._compute_gradient(self.scalar_potential)
            # Current density
            self.current_density = self.properties.conductivity * self.electric_field
        return {
            'vector_potential': self.vector_potential,
            'electric_field': self.electric_field,
            'magnetic_field': self.magnetic_field,
            'current_density': self.current_density,
            'frequency': frequency
        }
    def _apply_em_bc(self, bc: Dict[str, Any], matrix: sparse.spmatrix, rhs: np.ndarray):
        """Apply electromagnetic boundary conditions."""
        # Perfect electric conductor (PEC): n×E = 0
        if 'pec' in bc:
            for node in bc['pec']:
                # Zero tangential E-field
                for i in range(3):
                    idx = node * 3 + i
                    matrix[idx, idx] = 1.0
                    rhs[idx] = 0.0
        # Applied current density
        if 'current' in bc:
            for node, current in bc['current'].items():
                for i in range(3):
                    rhs[node * 3 + i] += current[i]
    def _apply_potential_bc(self, bc: Dict[int, float], matrix: sparse.spmatrix, rhs: np.ndarray):
        """Apply scalar potential boundary conditions."""
        for node, potential in bc.items():
            matrix[node, node] = 1.0
            rhs[node] = potential
    def _compute_curl(self, vector_field: np.ndarray) -> np.ndarray:
        """Compute curl of vector field (simplified)."""
        # In practice, use proper finite element curl operator
        n_nodes = len(vector_field)
        curl = np.zeros_like(vector_field)
        # Simplified finite difference approximation
        h = 0.01  # Grid spacing
        for i in range(1, n_nodes-1):
            # ∇×A = (∂Az/∂y - ∂Ay/∂z, ∂Ax/∂z - ∂Az/∂x, ∂Ay/∂x - ∂Ax/∂y)
            # Simplified for demonstration
            curl[i, 0] = (vector_field[i+1, 2] - vector_field[i-1, 2]) / (2*h)
            curl[i, 1] = -(vector_field[i+1, 0] - vector_field[i-1, 0]) / (2*h)
            curl[i, 2] = (vector_field[i+1, 1] - vector_field[i-1, 1]) / (2*h)
        return curl
    def _compute_gradient(self, scalar_field: np.ndarray) -> np.ndarray:
        """Compute gradient of scalar field."""
        n_nodes = len(scalar_field)
        gradient = np.zeros((n_nodes, 3))
        # Simplified finite difference
        h = 0.01
        for i in range(1, n_nodes-1):
            gradient[i, 0] = (scalar_field[i+1] - scalar_field[i-1]) / (2*h)
        return gradient
    def compute_power_density(self) -> np.ndarray:
        """Compute electromagnetic power density."""
        if self.electric_field is None or self.current_density is None:
            return np.zeros(len(self.mesh['nodes']))
        # Joule heating: P = J·E = σ|E|²
        if np.iscomplexobj(self.electric_field):
            # Time-average power for AC
            power = 0.5 * self.properties.conductivity * np.sum(
                np.abs(self.electric_field)**2, axis=1
            )
        else:
            # DC power
            power = np.sum(self.current_density * self.electric_field, axis=1)
        return power
class JouleHeating:
    """Joule heating model.
    Computes heat generation from electrical current flow.
    """
    def __init__(self):
        """Initialize Joule heating model."""
        self.efficiency = 1.0  # All electrical power becomes heat
    def compute_heating(self,
                       current_density: np.ndarray,
                       electric_field: np.ndarray,
                       conductivity: float) -> np.ndarray:
        """Compute Joule heating power density.
        Args:
            current_density: Current density J (A/m²)
            electric_field: Electric field E (V/m)
            conductivity: Electrical conductivity (S/m)
        Returns:
            Volumetric heat generation (W/m³)
        """
        # Joule heating: Q = J·E = σ|E|²
        if np.iscomplexobj(electric_field):
            # AC fields - time-average power
            heating = 0.5 * conductivity * np.sum(np.abs(electric_field)**2, axis=-1)
        else:
            # DC fields
            heating = np.sum(current_density * electric_field, axis=-1)
        return self.efficiency * heating
    def compute_heating_from_current(self,
                                   current: float,
                                   resistance: float) -> float:
        """Compute total Joule heating from current and resistance.
        Args:
            current: Total current (A)
            resistance: Total resistance (Ω)
        Returns:
            Total heating power (W)
        """
        return self.efficiency * current**2 * resistance
class InductionHeating:
    """Induction heating solver.
    Solves coupled electromagnetic-thermal problem for
    induction heating applications.
    """
    def __init__(self,
                 mesh: Dict[str, np.ndarray],
                 em_props: ElectromagneticProperties,
                 thermal_props: Any):
        """Initialize induction heating solver.
        Args:
            mesh: Finite element mesh
            em_props: Electromagnetic properties
            thermal_props: Thermal properties
        """
        self.mesh = mesh
        self.em_props = em_props
        self.thermal_props = thermal_props
        # EM solver
        self.em_solver = MaxwellSolver(mesh, em_props, formulation="A-phi")
        # Heating model
        self.joule_heating = JouleHeating()
        # Temperature field
        self.temperature = None
    def solve_induction_heating(self,
                              coil_current: Dict[str, Any],
                              frequency: float,
                              thermal_bc: Dict[str, Any],
                              time_span: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """Solve induction heating problem.
        Args:
            coil_current: Coil current specification
            frequency: Operating frequency (Hz)
            thermal_bc: Thermal boundary conditions
            time_span: Time interval for transient
        Returns:
            Solution fields
        """
        # Solve electromagnetic problem
        em_solution = self.em_solver.solve_frequency_domain(
            frequency, {'current': coil_current}
        )
        # Compute Joule heating
        heating_power = self.joule_heating.compute_heating(
            em_solution['current_density'],
            em_solution['electric_field'],
            self.em_props.conductivity
        )
        # Solve thermal problem
        n_nodes = len(self.mesh['nodes'])
        if time_span is None:
            # Steady-state thermal
            self.temperature = self._solve_steady_thermal(heating_power, thermal_bc)
        else:
            # Transient thermal
            self.temperature = self._solve_transient_thermal(
                heating_power, thermal_bc, time_span
            )
        return {
            'electromagnetic': em_solution,
            'temperature': self.temperature,
            'heating_power': heating_power,
            'frequency': frequency
        }
    def _solve_steady_thermal(self, heat_source: np.ndarray, bc: Dict[str, Any]) -> np.ndarray:
        """Solve steady-state heat conduction."""
        n_nodes = len(self.mesh['nodes'])
        # Heat equation: -k∇²T = Q
        # Simplified to linear system
        k_matrix = sparse.eye(n_nodes) * self.thermal_props.conductivity
        rhs = heat_source
        # Apply BC
        if 'temperature' in bc:
            for node, temp in bc['temperature'].items():
                k_matrix[node, node] = 1.0
                rhs[node] = temp
        # Solve
        temperature = sparse_linalg.spsolve(k_matrix, rhs)
        return temperature
    def _solve_transient_thermal(self, heat_source: np.ndarray, bc: Dict[str, Any],
                               time_span: Tuple[float, float]) -> List[np.ndarray]:
        """Solve transient heat conduction."""
        # Simplified implementation
        warnings.warn("Transient induction heating not fully implemented")
        # Return steady-state as placeholder
        steady_temp = self._solve_steady_thermal(heat_source, bc)
        return [steady_temp]
    def optimize_coil_design(self,
                           target_temperature: float,
                           constraints: Dict[str, Any]) -> Dict[str, float]:
        """Optimize induction coil design.
        Args:
            target_temperature: Desired temperature
            constraints: Design constraints
        Returns:
            Optimal coil parameters
        """
        # Optimization variables: current, frequency, coil geometry
        # Simplified optimization
        optimal_frequency = 10e3  # 10 kHz typical
        # Skin depth consideration
        skin_depth = self.em_props.skin_depth(optimal_frequency)
        # Current based on target power
        # P = I²R, where R depends on skin depth
        return {
            'frequency': optimal_frequency,
            'current': 100.0,  # Placeholder
            'skin_depth': skin_depth,
            'turns': 10
        }
class EddyCurrentSolver:
    """Eddy current analysis solver.
    Specialized solver for eddy current problems in
    conducting materials.
    """
    def __init__(self,
                 mesh: Dict[str, np.ndarray],
                 properties: ElectromagneticProperties):
        """Initialize eddy current solver.
        Args:
            mesh: Finite element mesh
            properties: Material properties
        """
        self.mesh = mesh
        self.properties = properties
        self.solver = MaxwellSolver(mesh, properties, formulation="A-phi")
    def analyze_eddy_currents(self,
                            applied_field: Callable,
                            frequency: float,
                            motion: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
        """Analyze eddy currents in conductor.
        Args:
            applied_field: Applied magnetic field function
            frequency: Field frequency (Hz)
            motion: Conductor motion specification
        Returns:
            Eddy current analysis results
        """
        # Compute skin depth
        skin_depth = self.properties.skin_depth(frequency)
        # Solve EM problem with applied field
        n_nodes = len(self.mesh['nodes'])
        # Applied field as boundary condition
        bc = {}
        for i, node_pos in enumerate(self.mesh['nodes']):
            b_applied = applied_field(node_pos)
            # Convert to vector potential BC (simplified)
        # Include motion if specified
        if motion is not None:
            velocity = motion.get('velocity', np.zeros(3))
            # Motional EMF: E = v × B
            # This modifies the governing equations
        # Solve
        em_solution = self.solver.solve_frequency_domain(frequency, bc)
        # Extract eddy currents
        eddy_currents = em_solution['current_density']
        # Compute forces
        forces = self._compute_electromagnetic_forces(
            eddy_currents, em_solution['magnetic_field']
        )
        # Compute losses
        power_loss = self.solver.compute_power_density()
        return {
            'eddy_currents': eddy_currents,
            'magnetic_field': em_solution['magnetic_field'],
            'forces': forces,
            'power_loss': power_loss,
            'skin_depth': skin_depth,
            'total_loss': np.sum(power_loss)
        }
    def _compute_electromagnetic_forces(self,
                                      current_density: np.ndarray,
                                      magnetic_field: np.ndarray) -> np.ndarray:
        """Compute Lorentz forces."""
        # F = J × B
        if current_density.shape != magnetic_field.shape:
            raise ValueError("Field dimensions mismatch")
        forces = np.cross(current_density, magnetic_field, axis=-1)
        return forces
    def compute_impedance(self, frequency: float, geometry: Dict[str, float]) -> complex:
        """Compute frequency-dependent impedance.
        Args:
            frequency: Operating frequency
            geometry: Conductor geometry
        Returns:
            Complex impedance
        """
        # Skin depth
        delta = self.properties.skin_depth(frequency)
        # For cylindrical conductor
        if 'radius' in geometry:
            r = geometry['radius']
            length = geometry.get('length', 1.0)
            if r > 3 * delta:
                # Thick conductor approximation
                # R_ac = R_dc * r/(2δ)
                r_dc = length / (np.pi * r**2 * self.properties.conductivity)
                r_ac = r_dc * r / (2 * delta)
                # Internal inductance
                l_int = self.properties.permeability * length / (8 * np.pi)
            else:
                # Thin conductor
                r_ac = length / (np.pi * r**2 * self.properties.conductivity)
                l_int = self.properties.permeability * length / (2 * np.pi)
            # Impedance
            z = r_ac + 1j * 2 * np.pi * frequency * l_int
        else:
            raise ValueError("Unsupported geometry")
        return z
class ElectromagneticThermalCoupling(CoupledSystem):
    """Main electromagnetic-thermal coupling solver.
    Coordinates EM and thermal solvers for coupled problems.
    """
    def __init__(self,
                 mesh: Dict[str, np.ndarray],
                 em_props: ElectromagneticProperties,
                 thermal_props: Any,
                 coupling_scheme: CouplingScheme = CouplingScheme.STAGGERED):
        """Initialize coupled EM-thermal solver.
        Args:
            mesh: Finite element mesh
            em_props: EM properties
            thermal_props: Thermal properties
            coupling_scheme: Coupling strategy
        """
        super().__init__("EMThermal", ["electromagnetic", "thermal"], coupling_scheme)
        self.mesh = mesh
        self.em_props = em_props
        self.thermal_props = thermal_props
        # Component solvers
        self.em_solver = MaxwellSolver(mesh, em_props)
        self.induction_solver = InductionHeating(mesh, em_props, thermal_props)
        self.setup_domains()
        self.setup_coupling()
    def setup_domains(self):
        """Setup EM and thermal domains."""
        self.add_domain_solver("electromagnetic", self.em_solver)
        # Thermal solver would be added here
    def setup_coupling(self):
        """Setup EM-thermal coupling."""
        # Coupling through Joule heating
        pass
    def solve_coupled_em_thermal(self,
                               em_source: Dict[str, Any],
                               thermal_bc: Dict[str, Any],
                               frequency: float = 0.0,
                               max_iterations: int = 10,
                               tolerance: float = 1e-3) -> Dict[str, Any]:
        """Solve coupled EM-thermal problem.
        Args:
            em_source: EM source specification
            thermal_bc: Thermal boundary conditions
            frequency: EM frequency
            max_iterations: Coupling iterations
            tolerance: Convergence tolerance
        Returns:
            Coupled solution
        """
        n_nodes = len(self.mesh['nodes'])
        # Initial temperature
        temperature = np.ones(n_nodes) * thermal_bc.get('ambient', 293.15)
        converged = False
        iteration = 0
        while not converged and iteration < max_iterations:
            iteration += 1
            prev_temperature = temperature.copy()
            # Update temperature-dependent properties
            # σ(T), k(T), etc.
            # Solve EM problem
            if frequency > 0:
                em_solution = self.em_solver.solve_frequency_domain(
                    frequency, em_source
                )
            else:
                em_solution = self.em_solver.solve_static(em_source)
            # Compute Joule heating
            heating = JouleHeating()
            heat_source = heating.compute_heating(
                em_solution['current_density'],
                em_solution['electric_field'],
                self.em_props.conductivity
            )
            # Solve thermal problem
            # Simplified steady-state
            k_matrix = sparse.eye(n_nodes) * self.thermal_props.conductivity
            rhs = heat_source
            # Apply thermal BC
            if 'temperature' in thermal_bc:
                for node, temp in thermal_bc['temperature'].items():
                    k_matrix[node, node] = 1.0
                    rhs[node] = temp
            # Convection BC
            if 'convection' in thermal_bc:
                h = thermal_bc['convection']['coefficient']
                t_inf = thermal_bc['convection']['temperature']
                # Add to matrix and RHS
                for node in thermal_bc['convection'].get('nodes', []):
                    k_matrix[node, node] += h
                    rhs[node] += h * t_inf
            temperature = sparse_linalg.spsolve(k_matrix, rhs)
            # Check convergence
            temp_change = np.linalg.norm(temperature - prev_temperature)
            if temp_change < tolerance:
                converged = True
        if not converged:
            warnings.warn(f"EM-thermal coupling did not converge after {iteration} iterations")
        return {
            'electromagnetic': em_solution,
            'temperature': temperature,
            'heating_power': heat_source,
            'iterations': iteration,
            'converged': converged
        }
# Utility functions
def electromagnetic_heating(geometry: Dict[str, float],
                          current: float,
                          frequency: float,
                          material: Dict[str, float]) -> Dict[str, float]:
    """Calculate electromagnetic heating for simple geometries.
    Args:
        geometry: Geometric parameters
        current: Applied current (A)
        frequency: Frequency (Hz)
        material: Material properties
    Returns:
        Heating analysis results
    """
    # Material properties
    sigma = material.get('conductivity', 1e7)  # S/m
    mu = material.get('permeability', MU_0)
    # Skin depth
    if frequency > 0:
        delta = 1.0 / np.sqrt(np.pi * frequency * mu * sigma)
    else:
        delta = float('inf')  # DC case
    results = {'skin_depth': delta, 'frequency': frequency}
    if 'wire' in geometry:
        # Cylindrical wire
        radius = geometry['radius']
        length = geometry['length']
        if frequency == 0 or radius < delta:
            # DC or thin wire
            resistance = length / (np.pi * radius**2 * sigma)
        else:
            # AC with skin effect
            # Effective area ≈ π * 2r * δ for r >> δ
            if radius > 3 * delta:
                effective_area = np.pi * 2 * radius * delta
            else:
                # Bessel function solution needed for exact result
                effective_area = np.pi * radius**2  # Approximation
            resistance = length / (effective_area * sigma)
        power = current**2 * resistance
        results.update({
            'resistance': resistance,
            'power': power,
            'power_density': power / (np.pi * radius**2 * length)
        })
    elif 'plate' in geometry:
        # Flat plate
        width = geometry['width']
        length = geometry['length']
        thickness = geometry['thickness']
        if frequency == 0 or thickness < 2 * delta:
            # DC or thin plate
            resistance = length / (width * thickness * sigma)
        else:
            # AC with skin effect (both sides)
            effective_thickness = 2 * delta
            resistance = length / (width * effective_thickness * sigma)
        power = current**2 * resistance
        results.update({
            'resistance': resistance,
            'power': power,
            'power_density': power / (width * length * thickness)
        })
    return results
def coupled_em_thermal(mesh: Dict[str, np.ndarray],
                      em_source: Dict[str, Any],
                      boundary_conditions: Dict[str, Any],
                      materials: Dict[str, Any],
                      analysis_type: str = "steady") -> Dict[str, np.ndarray]:
    """Solve coupled electromagnetic-thermal problem.
    Args:
        mesh: Finite element mesh
        em_source: EM excitation
        boundary_conditions: Combined BC
        materials: Material properties
        analysis_type: Analysis type (steady/transient)
    Returns:
        Coupled solution fields
    """
    # Extract properties
    em_props = ElectromagneticProperties(
        conductivity=materials.get('sigma', 1e7),
        permittivity=materials.get('epsilon', EPSILON_0),
        permeability=materials.get('mu', MU_0)
    )
    thermal_props = type('ThermalProps', (), {
        'conductivity': materials.get('k', 400),
        'specific_heat': materials.get('cp', 400),
        'density': materials.get('rho', 8900)
    })()
    # Create coupled solver
    solver = ElectromagneticThermalCoupling(
        mesh, em_props, thermal_props
    )
    # Separate BC
    em_bc = boundary_conditions.get('electromagnetic', {})
    thermal_bc = boundary_conditions.get('thermal', {})
    # Frequency
    frequency = em_source.get('frequency', 0.0)
    # Solve
    result = solver.solve_coupled_em_thermal(
        em_source, thermal_bc, frequency
    )
    return result