"""Thermal-Mechanical Coupling Module.
This module implements thermal-mechanical coupling for problems involving
heat transfer and structural deformation, including thermal expansion,
thermoelastic analysis, and coupled heat conduction.
Classes:
    ThermalMechanicalCoupling: Main coupled solver
    ThermalExpansion: Thermal expansion models
    ThermoelasticSolver: Coupled thermoelastic solver
    HeatGenerationModel: Heat generation from deformation
Functions:
    thermal_stress_analysis: Compute thermal stresses
    coupled_heat_conduction: Solve coupled heat transfer
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
class ThermalProperties:
    """Thermal material properties."""
    conductivity: float  # W/(m·K)
    specific_heat: float  # J/(kg·K)
    density: float  # kg/m³
    thermal_expansion: float  # 1/K
    reference_temperature: float = 293.15  # K
    @property
    def thermal_diffusivity(self) -> float:
        """Compute thermal diffusivity."""
        return self.conductivity / (self.density * self.specific_heat)
@dataclass
class MechanicalProperties:
    """Mechanical material properties."""
    youngs_modulus: float  # Pa
    poissons_ratio: float
    density: float  # kg/m³
    yield_stress: Optional[float] = None  # Pa
    @property
    def shear_modulus(self) -> float:
        """Compute shear modulus."""
        return self.youngs_modulus / (2 * (1 + self.poissons_ratio))
    @property
    def bulk_modulus(self) -> float:
        """Compute bulk modulus."""
        return self.youngs_modulus / (3 * (1 - 2 * self.poissons_ratio))
    @property
    def lame_lambda(self) -> float:
        """First Lamé parameter."""
        E, nu = self.youngs_modulus, self.poissons_ratio
        return E * nu / ((1 + nu) * (1 - 2 * nu))
    @property
    def lame_mu(self) -> float:
        """Second Lamé parameter (shear modulus)."""
        return self.shear_modulus
class ThermalExpansion:
    """Thermal expansion models.
    Provides various thermal expansion models including
    isotropic, orthotropic, and temperature-dependent expansions.
    """
    def __init__(self, expansion_type: str = "isotropic"):
        """Initialize thermal expansion model.
        Args:
            expansion_type: Type of thermal expansion model
        """
        self.expansion_type = expansion_type
        self.reference_temperature = 293.15  # K
    def compute_thermal_strain(self,
                             temperature: np.ndarray,
                             properties: ThermalProperties,
                             direction: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute thermal strain.
        Args:
            temperature: Temperature field
            properties: Material properties
            direction: Direction for anisotropic expansion
        Returns:
            Thermal strain tensor/components
        """
        delta_T = temperature - properties.reference_temperature
        if self.expansion_type == "isotropic":
            # Isotropic expansion: ε_th = α * ΔT * I
            alpha = properties.thermal_expansion
            if temperature.ndim == 0:
                # Scalar temperature
                strain = alpha * delta_T * np.eye(3)
            else:
                # Field of temperatures
                n_points = len(temperature)
                strain = np.zeros((n_points, 3, 3))
                for i in range(n_points):
                    strain[i] = alpha * delta_T[i] * np.eye(3)
        elif self.expansion_type == "orthotropic":
            # Orthotropic expansion
            # Requires directional expansion coefficients
            raise NotImplementedError("Orthotropic expansion not implemented")
        elif self.expansion_type == "temperature_dependent":
            # Temperature-dependent expansion coefficient
            alpha = self._temperature_dependent_alpha(temperature)
            if temperature.ndim == 0:
                strain = alpha * delta_T * np.eye(3)
            else:
                n_points = len(temperature)
                strain = np.zeros((n_points, 3, 3))
                for i in range(n_points):
                    strain[i] = alpha[i] * delta_T[i] * np.eye(3)
        else:
            raise ValueError(f"Unknown expansion type: {self.expansion_type}")
        return strain
    def _temperature_dependent_alpha(self, temperature: np.ndarray) -> np.ndarray:
        """Temperature-dependent expansion coefficient."""
        # Example: Linear variation
        alpha_0 = 1e-5  # 1/K at reference
        beta = 1e-8     # Temperature coefficient
        return alpha_0 + beta * (temperature - self.reference_temperature)
    def compute_thermal_stress(self,
                             temperature: np.ndarray,
                             thermal_props: ThermalProperties,
                             mech_props: MechanicalProperties,
                             constraint: str = "free") -> np.ndarray:
        """Compute thermal stress.
        Args:
            temperature: Temperature field
            thermal_props: Thermal properties
            mech_props: Mechanical properties
            constraint: Constraint type (free, constrained)
        Returns:
            Thermal stress tensor
        """
        # Thermal strain
        thermal_strain = self.compute_thermal_strain(temperature, thermal_props)
        if constraint == "free":
            # No stress if free to expand
            if temperature.ndim == 0:
                stress = np.zeros((3, 3))
            else:
                stress = np.zeros_like(thermal_strain)
        elif constraint == "constrained":
            # Fully constrained: σ = -E/(1-2ν) * α * ΔT * I
            E = mech_props.youngs_modulus
            nu = mech_props.poissons_ratio
            # Bulk modulus for hydrostatic stress
            K = E / (3 * (1 - 2 * nu))
            if temperature.ndim == 0:
                stress = -3 * K * thermal_strain
            else:
                stress = np.zeros_like(thermal_strain)
                for i in range(len(temperature)):
                    stress[i] = -3 * K * thermal_strain[i]
        else:
            raise ValueError(f"Unknown constraint type: {constraint}")
        return stress
class ThermoelasticSolver:
    """Coupled thermoelastic solver.
    Solves coupled heat conduction and elastic deformation
    with thermal expansion effects.
    """
    def __init__(self,
                 mesh: Dict[str, np.ndarray],
                 thermal_props: ThermalProperties,
                 mech_props: MechanicalProperties):
        """Initialize thermoelastic solver.
        Args:
            mesh: Finite element mesh
            thermal_props: Thermal properties
            mech_props: Mechanical properties
        """
        self.mesh = mesh
        self.thermal_props = thermal_props
        self.mech_props = mech_props
        # Solution fields
        self.temperature = None
        self.displacement = None
        # System matrices
        self.thermal_mass = None
        self.thermal_stiffness = None
        self.elastic_stiffness = None
        self.coupling_matrix = None
        # Thermal expansion model
        self.thermal_expansion = ThermalExpansion("isotropic")
        self._setup_system()
    def _setup_system(self):
        """Setup finite element system."""
        n_nodes = len(self.mesh['nodes'])
        # Initialize fields
        self.temperature = np.ones(n_nodes) * self.thermal_props.reference_temperature
        self.displacement = np.zeros((n_nodes, 2))  # 2D
        # Build system matrices (simplified)
        # Thermal system: C * dT/dt + K_T * T = Q
        self.thermal_mass = sparse.eye(n_nodes) * \
                          (self.thermal_props.density * self.thermal_props.specific_heat)
        self.thermal_stiffness = sparse.eye(n_nodes) * self.thermal_props.conductivity
        # Elastic system: K_E * u = F - K_TE * T
        n_dof = n_nodes * 2
        self.elastic_stiffness = sparse.eye(n_dof) * self.mech_props.youngs_modulus
        # Coupling matrix (thermal expansion contribution)
        # Simplified - in practice, proper FE assembly needed
        self.coupling_matrix = sparse.eye(n_dof) * \
                             (self.mech_props.youngs_modulus *
                              self.thermal_props.thermal_expansion)
    def solve_steady_state(self,
                         thermal_bc: Dict[str, Any],
                         mechanical_bc: Dict[str, Any],
                         heat_source: Optional[np.ndarray] = None,
                         body_force: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Solve steady-state thermoelastic problem.
        Args:
            thermal_bc: Thermal boundary conditions
            mechanical_bc: Mechanical boundary conditions
            heat_source: Volumetric heat source
            body_force: Body force vector
        Returns:
            Solution fields
        """
        n_nodes = len(self.mesh['nodes'])
        # Solve thermal problem first (if not coupled)
        # K_T * T = Q
        thermal_rhs = np.zeros(n_nodes)
        if heat_source is not None:
            thermal_rhs += heat_source
        # Apply thermal BC
        self._apply_thermal_bc(thermal_bc, self.thermal_stiffness, thermal_rhs)
        # Solve thermal system
        self.temperature = sparse_linalg.spsolve(self.thermal_stiffness, thermal_rhs)
        # Compute thermal strain/stress
        thermal_strain = self.thermal_expansion.compute_thermal_strain(
            self.temperature, self.thermal_props
        )
        # Solve mechanical problem with thermal loading
        # K_E * u = F - F_thermal
        n_dof = n_nodes * 2
        mech_rhs = np.zeros(n_dof)
        if body_force is not None:
            mech_rhs += body_force.flatten()
        # Add thermal load
        # F_thermal = B^T * D * ε_thermal (simplified)
        thermal_load = self._compute_thermal_load(thermal_strain)
        mech_rhs -= thermal_load
        # Apply mechanical BC
        self._apply_mechanical_bc(mechanical_bc, self.elastic_stiffness, mech_rhs)
        # Solve mechanical system
        disp_flat = sparse_linalg.spsolve(self.elastic_stiffness, mech_rhs)
        self.displacement = disp_flat.reshape((n_nodes, 2))
        # Compute stresses
        total_stress = self._compute_stress(self.displacement, thermal_strain)
        return {
            'temperature': self.temperature,
            'displacement': self.displacement,
            'stress': total_stress,
            'thermal_strain': thermal_strain
        }
    def solve_transient(self,
                       time_span: Tuple[float, float],
                       dt: float,
                       initial_temp: np.ndarray,
                       thermal_bc: Dict[str, Any],
                       mechanical_bc: Dict[str, Any]) -> Dict[str, List[np.ndarray]]:
        """Solve transient thermoelastic problem.
        Args:
            time_span: Time interval (t0, tf)
            dt: Time step
            initial_temp: Initial temperature
            thermal_bc: Thermal boundary conditions
            mechanical_bc: Mechanical boundary conditions
        Returns:
            Time history of solution fields
        """
        t0, tf = time_span
        n_steps = int((tf - t0) / dt)
        # Initialize
        self.temperature = initial_temp.copy()
        history = {
            'time': [],
            'temperature': [],
            'displacement': [],
            'stress': []
        }
        # Time stepping
        for step in range(n_steps):
            t = t0 + (step + 1) * dt
            # Solve thermal problem
            # C * (T^{n+1} - T^n)/dt + K_T * T^{n+1} = Q
            thermal_matrix = self.thermal_mass / dt + self.thermal_stiffness
            thermal_rhs = self.thermal_mass @ self.temperature / dt
            # Apply BC and solve
            self._apply_thermal_bc(thermal_bc, thermal_matrix, thermal_rhs)
            self.temperature = sparse_linalg.spsolve(thermal_matrix, thermal_rhs)
            # Solve mechanical problem (quasi-static)
            thermal_strain = self.thermal_expansion.compute_thermal_strain(
                self.temperature, self.thermal_props
            )
            # Similar to steady-state mechanical solve
            n_dof = len(self.mesh['nodes']) * 2
            mech_rhs = -self._compute_thermal_load(thermal_strain)
            self._apply_mechanical_bc(mechanical_bc, self.elastic_stiffness, mech_rhs)
            disp_flat = sparse_linalg.spsolve(self.elastic_stiffness, mech_rhs)
            self.displacement = disp_flat.reshape((-1, 2))
            # Compute stress
            stress = self._compute_stress(self.displacement, thermal_strain)
            # Store history
            history['time'].append(t)
            history['temperature'].append(self.temperature.copy())
            history['displacement'].append(self.displacement.copy())
            history['stress'].append(stress)
        return history
    def _apply_thermal_bc(self, bc: Dict[str, Any], matrix: sparse.spmatrix, rhs: np.ndarray):
        """Apply thermal boundary conditions."""
        # Dirichlet BC
        if 'temperature' in bc:
            for node, temp in bc['temperature'].items():
                # Zero out row and set diagonal
                matrix.data[matrix.indptr[node]:matrix.indptr[node+1]] = 0
                matrix[node, node] = 1.0
                rhs[node] = temp
        # Neumann BC (heat flux)
        if 'flux' in bc:
            for node, flux in bc['flux'].items():
                rhs[node] += flux
    def _apply_mechanical_bc(self, bc: Dict[str, Any], matrix: sparse.spmatrix, rhs: np.ndarray):
        """Apply mechanical boundary conditions."""
        n_nodes = len(self.mesh['nodes'])
        # Fixed displacement
        if 'fixed' in bc:
            for node in bc['fixed']:
                for dof in range(2):  # 2D
                    idx = node * 2 + dof
                    matrix.data[matrix.indptr[idx]:matrix.indptr[idx+1]] = 0
                    matrix[idx, idx] = 1.0
                    rhs[idx] = 0.0
        # Prescribed displacement
        if 'displacement' in bc:
            for node, disp in bc['displacement'].items():
                for dof in range(2):
                    idx = node * 2 + dof
                    matrix.data[matrix.indptr[idx]:matrix.indptr[idx+1]] = 0
                    matrix[idx, idx] = 1.0
                    rhs[idx] = disp[dof]
    def _compute_thermal_load(self, thermal_strain: np.ndarray) -> np.ndarray:
        """Compute equivalent thermal load vector."""
        n_nodes = len(self.mesh['nodes'])
        n_dof = n_nodes * 2
        thermal_load = np.zeros(n_dof)
        # Simplified - in practice, integrate over elements
        E = self.mech_props.youngs_modulus
        nu = self.mech_props.poissons_ratio
        alpha = self.thermal_props.thermal_expansion
        # For each node (simplified)
        for i in range(n_nodes):
            if thermal_strain.ndim == 3:  # Field of strain tensors
                eps_th = thermal_strain[i]
            else:
                eps_th = thermal_strain
            # Thermal force contribution (2D plane stress)
            factor = E * alpha / (1 - nu)
            thermal_load[2*i] = factor * eps_th[0, 0]  # x-component
            thermal_load[2*i+1] = factor * eps_th[1, 1]  # y-component
        return thermal_load
    def _compute_stress(self, displacement: np.ndarray,
                       thermal_strain: np.ndarray) -> np.ndarray:
        """Compute total stress (mechanical + thermal)."""
        n_nodes = len(displacement)
        stress = np.zeros((n_nodes, 3))  # σxx, σyy, σxy for 2D
        # Simplified stress calculation
        # σ = D * (ε_mech - ε_thermal)
        E = self.mech_props.youngs_modulus
        nu = self.mech_props.poissons_ratio
        # Plane stress elasticity matrix
        D = E / (1 - nu**2) * np.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1-nu)/2]
        ])
        # For each node (simplified - should be computed at integration points)
        for i in range(n_nodes):
            # Compute mechanical strain (simplified)
            # In practice, use proper strain-displacement matrix
            if i > 0:
                du_dx = (displacement[i, 0] - displacement[i-1, 0]) / 0.1  # Placeholder
            else:
                du_dx = 0
            mech_strain = np.array([du_dx, 0, 0])  # Simplified
            # Thermal strain contribution
            if thermal_strain.ndim == 3:
                eps_th = np.array([thermal_strain[i, 0, 0],
                                  thermal_strain[i, 1, 1], 0])
            else:
                eps_th = np.array([thermal_strain[0, 0],
                                  thermal_strain[1, 1], 0])
            # Total stress
            stress[i] = D @ (mech_strain - eps_th)
        return stress
class HeatGenerationModel:
    """Heat generation from mechanical processes.
    Models heat generation from plastic deformation,
    friction, and viscoelastic dissipation.
    """
    def __init__(self, generation_type: str = "plastic"):
        """Initialize heat generation model.
        Args:
            generation_type: Type of heat generation
        """
        self.generation_type = generation_type
        self.efficiency = 0.9  # Fraction of work converted to heat
    def compute_heat_generation(self,
                              stress: np.ndarray,
                              strain_rate: np.ndarray,
                              material_props: Any) -> np.ndarray:
        """Compute volumetric heat generation rate.
        Args:
            stress: Stress tensor
            strain_rate: Strain rate tensor
            material_props: Material properties
        Returns:
            Heat generation rate (W/m³)
        """
        if self.generation_type == "plastic":
            # Plastic dissipation: q = β * σ : ε̇_p
            # where β is Taylor-Quinney coefficient
            # Check for yielding
            von_mises = self._compute_von_mises_stress(stress)
            if hasattr(material_props, 'yield_stress'):
                yielding = von_mises > material_props.yield_stress
            else:
                yielding = False
            if yielding:
                # Plastic power
                power = np.sum(stress * strain_rate)
                heat_gen = self.efficiency * power
            else:
                heat_gen = 0.0
        elif self.generation_type == "viscous":
            # Viscous dissipation: q = τ : ε̇
            power = np.sum(stress * strain_rate)
            heat_gen = power  # All viscous work becomes heat
        elif self.generation_type == "friction":
            # Frictional heating: q = μ * σ_n * v_slip
            # Requires contact information
            warnings.warn("Frictional heating requires contact data")
            heat_gen = 0.0
        else:
            raise ValueError(f"Unknown generation type: {self.generation_type}")
        return heat_gen
    def _compute_von_mises_stress(self, stress: np.ndarray) -> float:
        """Compute von Mises equivalent stress."""
        if stress.shape == (3, 3):
            # 3D stress tensor
            s_xx, s_yy, s_zz = stress[0, 0], stress[1, 1], stress[2, 2]
            s_xy, s_xz, s_yz = stress[0, 1], stress[0, 2], stress[1, 2]
            von_mises = np.sqrt(0.5 * ((s_xx - s_yy)**2 +
                                      (s_yy - s_zz)**2 +
                                      (s_zz - s_xx)**2 +
                                      6 * (s_xy**2 + s_xz**2 + s_yz**2)))
        else:
            # 2D or simplified
            von_mises = np.sqrt(stress[0]**2 - stress[0]*stress[1] +
                               stress[1]**2 + 3*stress[2]**2)
        return von_mises
class ThermalMechanicalCoupling(CoupledSystem):
    """Main thermal-mechanical coupling solver.
    Coordinates thermal and mechanical solvers with
    appropriate coupling strategies.
    """
    def __init__(self,
                 mesh: Dict[str, np.ndarray],
                 thermal_props: ThermalProperties,
                 mech_props: MechanicalProperties,
                 coupling_scheme: CouplingScheme = CouplingScheme.STAGGERED):
        """Initialize coupled solver.
        Args:
            mesh: Finite element mesh
            thermal_props: Thermal properties
            mech_props: Mechanical properties
            coupling_scheme: Coupling strategy
        """
        super().__init__("ThermalMechanical", ["thermal", "mechanical"], coupling_scheme)
        self.mesh = mesh
        self.thermal_props = thermal_props
        self.mech_props = mech_props
        # Thermoelastic solver
        self.thermoelastic = ThermoelasticSolver(mesh, thermal_props, mech_props)
        # Heat generation model
        self.heat_generation = HeatGenerationModel("plastic")
        self.setup_domains()
        self.setup_coupling()
    def setup_domains(self):
        """Setup thermal and mechanical domains."""
        # For integrated solver, domains are handled by ThermoelasticSolver
        self.add_domain_solver("integrated", self.thermoelastic)
    def setup_coupling(self):
        """Setup thermal-mechanical coupling."""
        # Coupling is handled internally by ThermoelasticSolver
        pass
    def solve_coupled_problem(self,
                            time_span: Optional[Tuple[float, float]] = None,
                            thermal_bc: Dict[str, Any] = {},
                            mechanical_bc: Dict[str, Any] = {},
                            initial_temp: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Solve coupled thermal-mechanical problem.
        Args:
            time_span: Time interval for transient analysis
            thermal_bc: Thermal boundary conditions
            mechanical_bc: Mechanical boundary conditions
            initial_temp: Initial temperature field
        Returns:
            Solution data
        """
        if time_span is None:
            # Steady-state analysis
            return self.thermoelastic.solve_steady_state(
                thermal_bc, mechanical_bc
            )
        else:
            # Transient analysis
            if initial_temp is None:
                n_nodes = len(self.mesh['nodes'])
                initial_temp = np.ones(n_nodes) * self.thermal_props.reference_temperature
            dt = 0.1  # Default time step
            return self.thermoelastic.solve_transient(
                time_span, dt, initial_temp, thermal_bc, mechanical_bc
            )
# Utility functions
def thermal_stress_analysis(geometry: Dict[str, float],
                          temperature_field: Callable,
                          material: Dict[str, float],
                          constraint_type: str = "free") -> Dict[str, np.ndarray]:
    """Analyze thermal stresses in simple geometries.
    Args:
        geometry: Geometric parameters
        temperature_field: Temperature distribution function
        material: Material properties
        constraint_type: Constraint condition
    Returns:
        Stress analysis results
    """
    # Create simple mesh
    if 'plate' in geometry:
        # Rectangular plate
        L = geometry['length']
        W = geometry['width']
        nx, ny = 20, 10
        x = np.linspace(0, L, nx)
        y = np.linspace(0, W, ny)
        X, Y = np.meshgrid(x, y)
        nodes = np.column_stack([X.ravel(), Y.ravel()])
    else:
        raise ValueError("Unsupported geometry")
    # Evaluate temperature
    temperature = temperature_field(nodes[:, 0], nodes[:, 1])
    # Material properties
    thermal_props = ThermalProperties(
        conductivity=material.get('k', 50),
        specific_heat=material.get('cp', 500),
        density=material.get('rho', 7800),
        thermal_expansion=material.get('alpha', 1.2e-5)
    )
    mech_props = MechanicalProperties(
        youngs_modulus=material.get('E', 200e9),
        poissons_ratio=material.get('nu', 0.3),
        density=material.get('rho', 7800)
    )
    # Compute thermal stresses
    expansion_model = ThermalExpansion("isotropic")
    stress = expansion_model.compute_thermal_stress(
        temperature, thermal_props, mech_props, constraint_type
    )
    return {
        'nodes': nodes,
        'temperature': temperature,
        'stress': stress,
        'max_stress': np.max(np.abs(stress))
    }
def coupled_heat_conduction(mesh: Dict[str, np.ndarray],
                          thermal_props: ThermalProperties,
                          mech_props: MechanicalProperties,
                          heat_source: Callable,
                          boundary_conditions: Dict[str, Any],
                          coupling_effects: bool = True) -> Dict[str, np.ndarray]:
    """Solve coupled heat conduction with mechanical effects.
    Args:
        mesh: Finite element mesh
        thermal_props: Thermal properties
        mech_props: Mechanical properties
        heat_source: Heat source function
        boundary_conditions: Combined BC
        coupling_effects: Include coupling effects
    Returns:
        Coupled solution
    """
    # Create coupled solver
    coupled_solver = ThermalMechanicalCoupling(
        mesh, thermal_props, mech_props
    )
    # Separate boundary conditions
    thermal_bc = boundary_conditions.get('thermal', {})
    mechanical_bc = boundary_conditions.get('mechanical', {})
    # Solve
    if coupling_effects:
        result = coupled_solver.solve_coupled_problem(
            thermal_bc=thermal_bc,
            mechanical_bc=mechanical_bc
        )
    else:
        # Solve thermal only
        solver = ThermoelasticSolver(mesh, thermal_props, mech_props)
        result = solver.solve_steady_state(
            thermal_bc, mechanical_bc
        )
    return result