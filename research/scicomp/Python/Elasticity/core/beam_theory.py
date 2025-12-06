"""
Structural Beam Analysis and Theory
Comprehensive beam analysis including Euler-Bernoulli, Timoshenko, and
nonlinear beam theories with various boundary conditions and loading scenarios.
"""
import numpy as np
from typing import Tuple, Optional, Callable, List, Union
from dataclasses import dataclass
from scipy import integrate, optimize, sparse, linalg
from abc import ABC, abstractmethod
from .stress_strain import ElasticConstants
@dataclass
class BeamProperties:
    """Beam geometric and material properties."""
    length: float           # Beam length (m)
    elastic_modulus: float  # Young's modulus (Pa)
    area: float            # Cross-sectional area (m²)
    moment_of_inertia: float  # Second moment of area (m⁴)
    shear_modulus: Optional[float] = None  # Shear modulus (Pa)
    shear_correction: float = 5/6  # Shear correction factor
    density: float = 7850   # Material density (kg/m³)
    def __post_init__(self):
        """Calculate derived properties."""
        if self.shear_modulus is None:
            # Estimate from Young's modulus (assuming ν = 0.3)
            self.shear_modulus = self.elastic_modulus / (2 * (1 + 0.3))
class LoadCase:
    """Define loading conditions for beam analysis."""
    def __init__(self, load_type: str, **kwargs):
        """
        Initialize load case.
        Parameters:
            load_type: 'point', 'distributed', 'moment', or 'function'
            **kwargs: Load-specific parameters
        """
        self.load_type = load_type
        self.parameters = kwargs
    @classmethod
    def point_load(cls, force: float, position: float) -> 'LoadCase':
        """Create point load."""
        return cls('point', force=force, position=position)
    @classmethod
    def distributed_load(cls, intensity: Union[float, Callable],
                        start: float = 0, end: Optional[float] = None) -> 'LoadCase':
        """Create distributed load."""
        return cls('distributed', intensity=intensity, start=start, end=end)
    @classmethod
    def moment_load(cls, moment: float, position: float) -> 'LoadCase':
        """Create applied moment."""
        return cls('moment', moment=moment, position=position)
    @classmethod
    def function_load(cls, load_function: Callable) -> 'LoadCase':
        """Create load defined by function q(x)."""
        return cls('function', load_function=load_function)
    def evaluate_load(self, x: np.ndarray, beam_length: float) -> np.ndarray:
        """Evaluate load distribution at given positions."""
        if self.load_type == 'point':
            # Dirac delta approximation
            pos = self.parameters['position']
            force = self.parameters['force']
            load = np.zeros_like(x)
            idx = np.argmin(np.abs(x - pos))
            if idx < len(load):
                dx = x[1] - x[0] if len(x) > 1 else 1.0
                load[idx] = force / dx
            return load
        elif self.load_type == 'distributed':
            intensity = self.parameters['intensity']
            start = self.parameters['start']
            end = self.parameters.get('end', beam_length)
            if callable(intensity):
                load = np.array([intensity(xi) if start <= xi <= end else 0 for xi in x])
            else:
                load = np.where((x >= start) & (x <= end), intensity, 0)
            return load
        elif self.load_type == 'function':
            load_func = self.parameters['load_function']
            return np.array([load_func(xi) for xi in x])
        else:
            return np.zeros_like(x)
class BoundaryCondition:
    """Define boundary conditions for beam analysis."""
    def __init__(self, condition_type: str, position: str, value: float = 0):
        """
        Initialize boundary condition.
        Parameters:
            condition_type: 'displacement', 'slope', 'moment', 'shear'
            position: 'left' or 'right'
            value: Prescribed value
        """
        self.condition_type = condition_type
        self.position = position
        self.value = value
    @classmethod
    def simply_supported(cls) -> List['BoundaryCondition']:
        """Create simply supported boundary conditions."""
        return [
            cls('displacement', 'left', 0),
            cls('displacement', 'right', 0),
            cls('moment', 'left', 0),
            cls('moment', 'right', 0)
        ]
    @classmethod
    def clamped(cls) -> List['BoundaryCondition']:
        """Create clamped (fixed) boundary conditions."""
        return [
            cls('displacement', 'left', 0),
            cls('displacement', 'right', 0),
            cls('slope', 'left', 0),
            cls('slope', 'right', 0)
        ]
    @classmethod
    def cantilever(cls) -> List['BoundaryCondition']:
        """Create cantilever boundary conditions."""
        return [
            cls('displacement', 'left', 0),
            cls('slope', 'left', 0),
            cls('moment', 'right', 0),
            cls('shear', 'right', 0)
        ]
class EulerBernoulliBeam:
    """
    Euler-Bernoulli beam theory implementation.
    Features:
    - Linear and nonlinear analysis
    - Multiple boundary conditions
    - Static and dynamic analysis
    - Buckling analysis
    Examples:
        >>> beam = EulerBernoulliBeam(properties, boundary_conditions)
        >>> deflection = beam.static_analysis([point_load, distributed_load])
        >>> frequencies = beam.natural_frequencies(n_modes=5)
    """
    def __init__(self, properties: BeamProperties,
                 boundary_conditions: List[BoundaryCondition]):
        """Initialize Euler-Bernoulli beam."""
        self.properties = properties
        self.boundary_conditions = boundary_conditions
        # Derived properties
        self.EI = properties.elastic_modulus * properties.moment_of_inertia
        self.EA = properties.elastic_modulus * properties.area
        self.rho_A = properties.density * properties.area
    def static_analysis(self, loads: List[LoadCase], n_elements: int = 100) -> dict:
        """
        Perform static analysis using finite element method.
        Parameters:
            loads: List of load cases
            n_elements: Number of finite elements
        Returns:
            Dictionary with results
        """
        L = self.properties.length
        x = np.linspace(0, L, n_elements + 1)
        dx = L / n_elements
        # Assemble stiffness matrix
        K = self._assemble_stiffness_matrix(n_elements)
        # Assemble load vector
        F = self._assemble_load_vector(loads, x)
        # Apply boundary conditions
        K_bc, F_bc = self._apply_boundary_conditions(K, F, x)
        # Solve system
        u = linalg.solve(K_bc, F_bc)
        # Calculate derived quantities
        deflection = u[::2]  # Extract deflection DOFs
        slope = u[1::2]      # Extract slope DOFs
        # Calculate moment and shear
        moment = self._calculate_moment(u, x)
        shear = self._calculate_shear(u, x)
        return {
            'x': x,
            'deflection': deflection,
            'slope': slope,
            'moment': moment,
            'shear': shear,
            'max_deflection': np.max(np.abs(deflection)),
            'max_moment': np.max(np.abs(moment))
        }
    def _assemble_stiffness_matrix(self, n_elements: int) -> np.ndarray:
        """Assemble global stiffness matrix."""
        n_nodes = n_elements + 1
        n_dof = 2 * n_nodes  # 2 DOF per node (deflection, slope)
        K_global = np.zeros((n_dof, n_dof))
        L_e = self.properties.length / n_elements  # Element length
        # Element stiffness matrix for Euler-Bernoulli beam
        K_element = (self.EI / L_e**3) * np.array([
            [12, 6*L_e, -12, 6*L_e],
            [6*L_e, 4*L_e**2, -6*L_e, 2*L_e**2],
            [-12, -6*L_e, 12, -6*L_e],
            [6*L_e, 2*L_e**2, -6*L_e, 4*L_e**2]
        ])
        # Assemble global matrix
        for e in range(n_elements):
            # Global DOF indices for element
            dof_indices = [2*e, 2*e+1, 2*e+2, 2*e+3]
            # Add element contribution
            for i in range(4):
                for j in range(4):
                    K_global[dof_indices[i], dof_indices[j]] += K_element[i, j]
        return K_global
    def _assemble_load_vector(self, loads: List[LoadCase], x: np.ndarray) -> np.ndarray:
        """Assemble global load vector."""
        n_nodes = len(x)
        F = np.zeros(2 * n_nodes)
        for load in loads:
            if load.load_type == 'point':
                # Point load
                pos = load.parameters['position']
                force = load.parameters['force']
                idx = np.argmin(np.abs(x - pos))
                F[2*idx] += force  # Add to deflection DOF
            elif load.load_type == 'distributed':
                # Distributed load - use consistent load vector
                load_values = load.evaluate_load(x, self.properties.length)
                dx = x[1] - x[0]
                # Trapezoidal integration for consistent load vector
                for i in range(len(x)):
                    F[2*i] += load_values[i] * dx
            elif load.load_type == 'moment':
                # Applied moment
                pos = load.parameters['position']
                moment = load.parameters['moment']
                idx = np.argmin(np.abs(x - pos))
                F[2*idx+1] += moment  # Add to slope DOF
        return F
    def _apply_boundary_conditions(self, K: np.ndarray, F: np.ndarray,
                                 x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply boundary conditions using penalty method."""
        K_bc = K.copy()
        F_bc = F.copy()
        penalty = 1e12 * np.max(np.diag(K))
        for bc in self.boundary_conditions:
            if bc.position == 'left':
                node_idx = 0
            else:  # right
                node_idx = len(x) - 1
            if bc.condition_type == 'displacement':
                dof_idx = 2 * node_idx
                K_bc[dof_idx, dof_idx] += penalty
                F_bc[dof_idx] += penalty * bc.value
            elif bc.condition_type == 'slope':
                dof_idx = 2 * node_idx + 1
                K_bc[dof_idx, dof_idx] += penalty
                F_bc[dof_idx] += penalty * bc.value
            # Moment and shear BCs would need special treatment
        return K_bc, F_bc
    def _calculate_moment(self, u: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Calculate bending moment from displacement field."""
        deflection = u[::2]
        # M = -EI d²w/dx²
        d2w_dx2 = np.gradient(np.gradient(deflection, x), x)
        return -self.EI * d2w_dx2
    def _calculate_shear(self, u: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Calculate shear force from displacement field."""
        deflection = u[::2]
        # V = -EI d³w/dx³
        d3w_dx3 = np.gradient(np.gradient(np.gradient(deflection, x), x), x)
        return -self.EI * d3w_dx3
    def natural_frequencies(self, n_modes: int = 10) -> np.ndarray:
        """
        Calculate natural frequencies using finite element method.
        Parameters:
            n_modes: Number of modes to calculate
        Returns:
            Natural frequencies (Hz)
        """
        n_elements = 50
        # Assemble stiffness and mass matrices
        K = self._assemble_stiffness_matrix(n_elements)
        M = self._assemble_mass_matrix(n_elements)
        # Apply boundary conditions to both matrices
        x = np.linspace(0, self.properties.length, n_elements + 1)
        K_bc, _ = self._apply_boundary_conditions(K, np.zeros(K.shape[0]), x)
        M_bc, _ = self._apply_boundary_conditions(M, np.zeros(M.shape[0]), x)
        # Solve generalized eigenvalue problem
        eigenvals, eigenvecs = linalg.eigh(K_bc, M_bc)
        # Convert to frequencies
        frequencies = np.sqrt(np.maximum(eigenvals, 0)) / (2 * np.pi)
        return frequencies[:n_modes]
    def _assemble_mass_matrix(self, n_elements: int) -> np.ndarray:
        """Assemble global mass matrix."""
        n_nodes = n_elements + 1
        n_dof = 2 * n_nodes
        M_global = np.zeros((n_dof, n_dof))
        L_e = self.properties.length / n_elements
        # Consistent mass matrix for Euler-Bernoulli beam element
        M_element = (self.rho_A * L_e / 420) * np.array([
            [156, 22*L_e, 54, -13*L_e],
            [22*L_e, 4*L_e**2, 13*L_e, -3*L_e**2],
            [54, 13*L_e, 156, -22*L_e],
            [-13*L_e, -3*L_e**2, -22*L_e, 4*L_e**2]
        ])
        # Assemble global matrix
        for e in range(n_elements):
            dof_indices = [2*e, 2*e+1, 2*e+2, 2*e+3]
            for i in range(4):
                for j in range(4):
                    M_global[dof_indices[i], dof_indices[j]] += M_element[i, j]
        return M_global
    def buckling_analysis(self, axial_load: float) -> float:
        """
        Perform linear buckling analysis.
        Parameters:
            axial_load: Applied axial load (N)
        Returns:
            Critical buckling load factor
        """
        n_elements = 50
        # Assemble stiffness matrices
        K = self._assemble_stiffness_matrix(n_elements)
        K_geo = self._assemble_geometric_stiffness_matrix(n_elements, axial_load)
        # Apply boundary conditions
        x = np.linspace(0, self.properties.length, n_elements + 1)
        K_bc, _ = self._apply_boundary_conditions(K, np.zeros(K.shape[0]), x)
        K_geo_bc, _ = self._apply_boundary_conditions(K_geo, np.zeros(K_geo.shape[0]), x)
        # Solve eigenvalue problem (K + λ*K_geo) = 0
        eigenvals, _ = linalg.eigh(-K_geo_bc, K_bc)
        # First positive eigenvalue is critical load factor
        positive_eigenvals = eigenvals[eigenvals > 0]
        if len(positive_eigenvals) > 0:
            return positive_eigenvals[0]
        else:
            return np.inf
    def _assemble_geometric_stiffness_matrix(self, n_elements: int,
                                           axial_load: float) -> np.ndarray:
        """Assemble geometric stiffness matrix for buckling analysis."""
        n_nodes = n_elements + 1
        n_dof = 2 * n_nodes
        K_geo = np.zeros((n_dof, n_dof))
        L_e = self.properties.length / n_elements
        P = axial_load
        # Geometric stiffness matrix for beam element
        K_geo_element = (P / (30 * L_e)) * np.array([
            [36, 3*L_e, -36, 3*L_e],
            [3*L_e, 4*L_e**2, -3*L_e, -L_e**2],
            [-36, -3*L_e, 36, -3*L_e],
            [3*L_e, -L_e**2, -3*L_e, 4*L_e**2]
        ])
        # Assemble global matrix
        for e in range(n_elements):
            dof_indices = [2*e, 2*e+1, 2*e+2, 2*e+3]
            for i in range(4):
                for j in range(4):
                    K_geo[dof_indices[i], dof_indices[j]] += K_geo_element[i, j]
        return K_geo
class TimoshenkoBeam:
    """
    Timoshenko beam theory including shear deformation effects.
    Features:
    - Shear deformation effects
    - Thick beam analysis
    - Improved accuracy for short beams
    - Dynamic analysis with rotatory inertia
    """
    def __init__(self, properties: BeamProperties,
                 boundary_conditions: List[BoundaryCondition]):
        """Initialize Timoshenko beam."""
        self.properties = properties
        self.boundary_conditions = boundary_conditions
        # Derived properties
        self.EI = properties.elastic_modulus * properties.moment_of_inertia
        self.kGA = properties.shear_correction * properties.shear_modulus * properties.area
        self.rho_A = properties.density * properties.area
        # Rotatory inertia per unit length
        self.rho_I = properties.density * properties.moment_of_inertia
    def static_analysis(self, loads: List[LoadCase], n_elements: int = 100) -> dict:
        """Perform static analysis using Timoshenko beam theory."""
        L = self.properties.length
        x = np.linspace(0, L, n_elements + 1)
        # Assemble stiffness matrix (includes shear deformation)
        K = self._assemble_timoshenko_stiffness_matrix(n_elements)
        # Assemble load vector
        F = self._assemble_load_vector(loads, x)
        # Apply boundary conditions
        K_bc, F_bc = self._apply_boundary_conditions(K, F, x)
        # Solve system
        u = linalg.solve(K_bc, F_bc)
        # Extract results
        deflection = u[::2]  # Deflection w
        rotation = u[1::2]   # Rotation φ
        # Calculate shear strain and moment
        shear_strain = self._calculate_shear_strain(u, x)
        moment = self._calculate_moment_timoshenko(u, x)
        shear_force = self._calculate_shear_force_timoshenko(u, x)
        return {
            'x': x,
            'deflection': deflection,
            'rotation': rotation,
            'shear_strain': shear_strain,
            'moment': moment,
            'shear_force': shear_force,
            'max_deflection': np.max(np.abs(deflection))
        }
    def _assemble_timoshenko_stiffness_matrix(self, n_elements: int) -> np.ndarray:
        """Assemble Timoshenko beam stiffness matrix."""
        n_nodes = n_elements + 1
        n_dof = 2 * n_nodes  # w and φ at each node
        K_global = np.zeros((n_dof, n_dof))
        L_e = self.properties.length / n_elements
        # Timoshenko beam element stiffness matrix
        # [K] = [K_b] + [K_s] (bending + shear contributions)
        # Bending stiffness
        K_bending = (self.EI / L_e) * np.array([
            [0, 0, 0, 0],
            [0, 1, 0, -1],
            [0, 0, 0, 0],
            [0, -1, 0, 1]
        ])
        # Shear stiffness
        phi = 12 * self.EI / (self.kGA * L_e**2)  # Shear parameter
        K_shear = (self.kGA / ((1 + phi) * L_e)) * np.array([
            [1, L_e/2, -1, L_e/2],
            [L_e/2, L_e**2/4 + phi/3, -L_e/2, L_e**2/4 - phi/6],
            [-1, -L_e/2, 1, -L_e/2],
            [L_e/2, L_e**2/4 - phi/6, -L_e/2, L_e**2/4 + phi/3]
        ])
        K_element = K_bending + K_shear
        # Assemble global matrix
        for e in range(n_elements):
            dof_indices = [2*e, 2*e+1, 2*e+2, 2*e+3]
            for i in range(4):
                for j in range(4):
                    K_global[dof_indices[i], dof_indices[j]] += K_element[i, j]
        return K_global
    def _calculate_shear_strain(self, u: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Calculate shear strain γ = dw/dx - φ."""
        deflection = u[::2]
        rotation = u[1::2]
        dw_dx = np.gradient(deflection, x)
        shear_strain = dw_dx - rotation
        return shear_strain
    def _calculate_moment_timoshenko(self, u: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Calculate bending moment M = EI dφ/dx."""
        rotation = u[1::2]
        dphi_dx = np.gradient(rotation, x)
        return self.EI * dphi_dx
    def _calculate_shear_force_timoshenko(self, u: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Calculate shear force V = kGA γ."""
        shear_strain = self._calculate_shear_strain(u, x)
        return self.kGA * shear_strain
    def natural_frequencies_timoshenko(self, n_modes: int = 10) -> np.ndarray:
        """Calculate natural frequencies including rotatory inertia effects."""
        n_elements = 50
        # Assemble matrices
        K = self._assemble_timoshenko_stiffness_matrix(n_elements)
        M = self._assemble_timoshenko_mass_matrix(n_elements)
        # Apply boundary conditions
        x = np.linspace(0, self.properties.length, n_elements + 1)
        K_bc, _ = self._apply_boundary_conditions(K, np.zeros(K.shape[0]), x)
        M_bc, _ = self._apply_boundary_conditions(M, np.zeros(M.shape[0]), x)
        # Solve eigenvalue problem
        eigenvals, _ = linalg.eigh(K_bc, M_bc)
        # Convert to frequencies
        frequencies = np.sqrt(np.maximum(eigenvals, 0)) / (2 * np.pi)
        return frequencies[:n_modes]
    def _assemble_timoshenko_mass_matrix(self, n_elements: int) -> np.ndarray:
        """Assemble Timoshenko beam mass matrix with rotatory inertia."""
        n_nodes = n_elements + 1
        n_dof = 2 * n_nodes
        M_global = np.zeros((n_dof, n_dof))
        L_e = self.properties.length / n_elements
        # Mass matrix including rotatory inertia
        M_element = (L_e / 420) * np.array([
            [156*self.rho_A, 22*L_e*self.rho_A, 54*self.rho_A, -13*L_e*self.rho_A],
            [22*L_e*self.rho_A, (4*L_e**2*self.rho_A + 420*self.rho_I),
             13*L_e*self.rho_A, (-3*L_e**2*self.rho_A + 420*self.rho_I)],
            [54*self.rho_A, 13*L_e*self.rho_A, 156*self.rho_A, -22*L_e*self.rho_A],
            [-13*L_e*self.rho_A, (-3*L_e**2*self.rho_A + 420*self.rho_I),
             -22*L_e*self.rho_A, (4*L_e**2*self.rho_A + 420*self.rho_I)]
        ])
        # Assemble global matrix
        for e in range(n_elements):
            dof_indices = [2*e, 2*e+1, 2*e+2, 2*e+3]
            for i in range(4):
                for j in range(4):
                    M_global[dof_indices[i], dof_indices[j]] += M_element[i, j]
        return M_global
    # Inherit other methods from EulerBernoulliBeam
    def _assemble_load_vector(self, loads: List[LoadCase], x: np.ndarray) -> np.ndarray:
        """Reuse load vector assembly from Euler-Bernoulli."""
        # Same implementation as Euler-Bernoulli beam
        n_nodes = len(x)
        F = np.zeros(2 * n_nodes)
        for load in loads:
            if load.load_type == 'point':
                pos = load.parameters['position']
                force = load.parameters['force']
                idx = np.argmin(np.abs(x - pos))
                F[2*idx] += force
            elif load.load_type == 'distributed':
                load_values = load.evaluate_load(x, self.properties.length)
                dx = x[1] - x[0]
                for i in range(len(x)):
                    F[2*i] += load_values[i] * dx
        return F
    def _apply_boundary_conditions(self, K: np.ndarray, F: np.ndarray,
                                 x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply boundary conditions using penalty method."""
        K_bc = K.copy()
        F_bc = F.copy()
        penalty = 1e12 * np.max(np.diag(K))
        for bc in self.boundary_conditions:
            if bc.position == 'left':
                node_idx = 0
            else:
                node_idx = len(x) - 1
            if bc.condition_type == 'displacement':
                dof_idx = 2 * node_idx
                K_bc[dof_idx, dof_idx] += penalty
                F_bc[dof_idx] += penalty * bc.value
            elif bc.condition_type == 'slope':
                dof_idx = 2 * node_idx + 1
                K_bc[dof_idx, dof_idx] += penalty
                F_bc[dof_idx] += penalty * bc.value
        return K_bc, F_bc
def create_standard_beam_sections() -> dict:
    """Create database of standard beam cross-sections."""
    sections = {}
    # Rectangular section
    def rectangular_properties(width: float, height: float) -> dict:
        area = width * height
        I = width * height**3 / 12
        return {'area': area, 'moment_of_inertia': I, 'section_modulus': I / (height/2)}
    # I-beam section
    def i_beam_properties(width: float, height: float,
                         web_thickness: float, flange_thickness: float) -> dict:
        # Simplified I-beam calculation
        area = 2 * width * flange_thickness + (height - 2*flange_thickness) * web_thickness
        # Moment of inertia calculation for I-beam
        I_web = web_thickness * (height - 2*flange_thickness)**3 / 12
        I_flanges = 2 * (width * flange_thickness**3 / 12 +
                        width * flange_thickness * ((height - flange_thickness)/2)**2)
        I = I_web + I_flanges
        return {'area': area, 'moment_of_inertia': I, 'section_modulus': I / (height/2)}
    # Circular section
    def circular_properties(diameter: float) -> dict:
        area = np.pi * diameter**2 / 4
        I = np.pi * diameter**4 / 64
        return {'area': area, 'moment_of_inertia': I, 'section_modulus': I / (diameter/2)}
    sections['rectangular'] = rectangular_properties
    sections['i_beam'] = i_beam_properties
    sections['circular'] = circular_properties
    return sections