"""
Stress-Strain Analysis for Elastic Materials
Comprehensive stress and strain tensor operations, elastic moduli calculations,
and constitutive relationship modeling for isotropic and anisotropic materials.
"""
import numpy as np
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass
import warnings
from ..utils.constants import ELASTIC_CONSTANTS
@dataclass
class ElasticConstants:
    """Elastic constants for materials."""
    youngs_modulus: float  # E (Pa)
    poissons_ratio: float  # ν (dimensionless)
    shear_modulus: Optional[float] = None  # G (Pa)
    bulk_modulus: Optional[float] = None   # K (Pa)
    lame_first: Optional[float] = None     # λ (Pa)
    lame_second: Optional[float] = None    # μ (Pa)
    def __post_init__(self):
        """Calculate derived elastic constants."""
        E, nu = self.youngs_modulus, self.poissons_ratio
        # Validate input ranges
        if E <= 0:
            raise ValueError("Young's modulus must be positive")
        if not -1 < nu < 0.5:
            raise ValueError("Poisson's ratio must be in range (-1, 0.5)")
        # Calculate derived constants
        if self.shear_modulus is None:
            self.shear_modulus = E / (2 * (1 + nu))
        if self.bulk_modulus is None:
            self.bulk_modulus = E / (3 * (1 - 2 * nu))
        if self.lame_first is None:
            self.lame_first = (E * nu) / ((1 + nu) * (1 - 2 * nu))
        if self.lame_second is None:
            self.lame_second = self.shear_modulus
class StressTensor:
    """
    3D stress tensor representation and operations.
    Features:
    - Stress tensor creation and manipulation
    - Principal stress calculation
    - Invariant computation
    - Coordinate transformations
    - Von Mises stress calculation
    Examples:
        >>> stress = StressTensor(sigma_xx=100e6, sigma_yy=50e6, sigma_zz=25e6)
        >>> principal_stresses = stress.principal_stresses()
        >>> von_mises = stress.von_mises_stress()
    """
    def __init__(self, sigma_xx: float = 0, sigma_yy: float = 0, sigma_zz: float = 0,
                 sigma_xy: float = 0, sigma_xz: float = 0, sigma_yz: float = 0):
        """
        Initialize stress tensor.
        Parameters:
            sigma_xx, sigma_yy, sigma_zz: Normal stress components (Pa)
            sigma_xy, sigma_xz, sigma_yz: Shear stress components (Pa)
        """
        self.tensor = np.array([
            [sigma_xx, sigma_xy, sigma_xz],
            [sigma_xy, sigma_yy, sigma_yz],
            [sigma_xz, sigma_yz, sigma_zz]
        ])
    @classmethod
    def from_array(cls, tensor: np.ndarray) -> 'StressTensor':
        """Create stress tensor from 3x3 array."""
        if tensor.shape != (3, 3):
            raise ValueError("Stress tensor must be 3x3 array")
        stress = cls()
        stress.tensor = tensor.copy()
        return stress
    @classmethod
    def from_voigt(cls, voigt: np.ndarray) -> 'StressTensor':
        """Create stress tensor from Voigt notation [σxx, σyy, σzz, σyz, σxz, σxy]."""
        if len(voigt) != 6:
            raise ValueError("Voigt notation must have 6 components")
        return cls(
            sigma_xx=voigt[0], sigma_yy=voigt[1], sigma_zz=voigt[2],
            sigma_yz=voigt[3], sigma_xz=voigt[4], sigma_xy=voigt[5]
        )
    def to_voigt(self) -> np.ndarray:
        """Convert to Voigt notation."""
        return np.array([
            self.tensor[0, 0],  # σxx
            self.tensor[1, 1],  # σyy
            self.tensor[2, 2],  # σzz
            self.tensor[1, 2],  # σyz
            self.tensor[0, 2],  # σxz
            self.tensor[0, 1]   # σxy
        ])
    def principal_stresses(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate principal stresses and directions.
        Returns:
            Tuple of (principal_values, principal_directions)
        """
        eigenvalues, eigenvectors = np.linalg.eigh(self.tensor)
        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        principal_values = eigenvalues[idx]
        principal_directions = eigenvectors[:, idx]
        return principal_values, principal_directions
    def invariants(self) -> Tuple[float, float, float]:
        """
        Calculate stress tensor invariants.
        Returns:
            Tuple of (I1, I2, I3) stress invariants
        """
        s = self.tensor
        I1 = np.trace(s)
        I2 = 0.5 * (np.trace(s)**2 - np.trace(s @ s))
        I3 = np.linalg.det(s)
        return I1, I2, I3
    def deviatoric(self) -> 'StressTensor':
        """Calculate deviatoric stress tensor."""
        mean_stress = np.trace(self.tensor) / 3
        deviatoric_tensor = self.tensor - mean_stress * np.eye(3)
        return StressTensor.from_array(deviatoric_tensor)
    def von_mises_stress(self) -> float:
        """Calculate von Mises equivalent stress."""
        dev = self.deviatoric()
        return np.sqrt(1.5 * np.sum(dev.tensor * dev.tensor))
    def maximum_shear_stress(self) -> float:
        """Calculate maximum shear stress."""
        principal_stresses, _ = self.principal_stresses()
        return (principal_stresses[0] - principal_stresses[2]) / 2
    def octahedral_shear_stress(self) -> float:
        """Calculate octahedral shear stress."""
        principal_stresses, _ = self.principal_stresses()
        s1, s2, s3 = principal_stresses
        return np.sqrt((s1 - s2)**2 + (s2 - s3)**2 + (s3 - s1)**2) / 3
    def transform(self, rotation_matrix: np.ndarray) -> 'StressTensor':
        """Transform stress tensor to new coordinate system."""
        if rotation_matrix.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3")
        R = rotation_matrix
        transformed_tensor = R @ self.tensor @ R.T
        return StressTensor.from_array(transformed_tensor)
    def __add__(self, other: 'StressTensor') -> 'StressTensor':
        """Add stress tensors."""
        return StressTensor.from_array(self.tensor + other.tensor)
    def __sub__(self, other: 'StressTensor') -> 'StressTensor':
        """Subtract stress tensors."""
        return StressTensor.from_array(self.tensor - other.tensor)
    def __mul__(self, scalar: float) -> 'StressTensor':
        """Multiply stress tensor by scalar."""
        return StressTensor.from_array(scalar * self.tensor)
    def __str__(self) -> str:
        """String representation."""
        return f"StressTensor:\n{self.tensor}"
class StrainTensor:
    """
    3D strain tensor representation and operations.
    Features:
    - Strain tensor creation and manipulation
    - Principal strain calculation
    - Strain invariants
    - Compatibility checks
    - Finite vs infinitesimal strain
    Examples:
        >>> strain = StrainTensor(epsilon_xx=0.001, epsilon_yy=0.0005)
        >>> principal_strains = strain.principal_strains()
        >>> volumetric_strain = strain.volumetric_strain()
    """
    def __init__(self, epsilon_xx: float = 0, epsilon_yy: float = 0, epsilon_zz: float = 0,
                 gamma_xy: float = 0, gamma_xz: float = 0, gamma_yz: float = 0,
                 engineering_strain: bool = True):
        """
        Initialize strain tensor.
        Parameters:
            epsilon_xx, epsilon_yy, epsilon_zz: Normal strain components
            gamma_xy, gamma_xz, gamma_yz: Shear strain components
            engineering_strain: If True, use engineering shear strains (γ = 2ε)
        """
        # Convert engineering shear strains to tensor shear strains
        shear_factor = 0.5 if engineering_strain else 1.0
        self.tensor = np.array([
            [epsilon_xx, shear_factor * gamma_xy, shear_factor * gamma_xz],
            [shear_factor * gamma_xy, epsilon_yy, shear_factor * gamma_yz],
            [shear_factor * gamma_xz, shear_factor * gamma_yz, epsilon_zz]
        ])
    @classmethod
    def from_array(cls, tensor: np.ndarray) -> 'StrainTensor':
        """Create strain tensor from 3x3 array."""
        if tensor.shape != (3, 3):
            raise ValueError("Strain tensor must be 3x3 array")
        strain = cls()
        strain.tensor = tensor.copy()
        return strain
    @classmethod
    def from_voigt(cls, voigt: np.ndarray, engineering_strain: bool = True) -> 'StrainTensor':
        """Create strain tensor from Voigt notation."""
        if len(voigt) != 6:
            raise ValueError("Voigt notation must have 6 components")
        return cls(
            epsilon_xx=voigt[0], epsilon_yy=voigt[1], epsilon_zz=voigt[2],
            gamma_yz=voigt[3], gamma_xz=voigt[4], gamma_xy=voigt[5],
            engineering_strain=engineering_strain
        )
    def to_voigt(self, engineering_strain: bool = True) -> np.ndarray:
        """Convert to Voigt notation."""
        shear_factor = 2.0 if engineering_strain else 1.0
        return np.array([
            self.tensor[0, 0],                    # εxx
            self.tensor[1, 1],                    # εyy
            self.tensor[2, 2],                    # εzz
            shear_factor * self.tensor[1, 2],     # γyz
            shear_factor * self.tensor[0, 2],     # γxz
            shear_factor * self.tensor[0, 1]      # γxy
        ])
    def principal_strains(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate principal strains and directions."""
        eigenvalues, eigenvectors = np.linalg.eigh(self.tensor)
        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        principal_values = eigenvalues[idx]
        principal_directions = eigenvectors[:, idx]
        return principal_values, principal_directions
    def volumetric_strain(self) -> float:
        """Calculate volumetric strain (trace of strain tensor)."""
        return np.trace(self.tensor)
    def deviatoric(self) -> 'StrainTensor':
        """Calculate deviatoric strain tensor."""
        volumetric_strain = self.volumetric_strain() / 3
        deviatoric_tensor = self.tensor - volumetric_strain * np.eye(3)
        return StrainTensor.from_array(deviatoric_tensor)
    def equivalent_strain(self) -> float:
        """Calculate equivalent strain (von Mises equivalent)."""
        dev = self.deviatoric()
        return np.sqrt(2.0/3.0 * np.sum(dev.tensor * dev.tensor))
    def maximum_shear_strain(self) -> float:
        """Calculate maximum shear strain."""
        principal_strains, _ = self.principal_strains()
        return (principal_strains[0] - principal_strains[2]) / 2
    def transform(self, rotation_matrix: np.ndarray) -> 'StrainTensor':
        """Transform strain tensor to new coordinate system."""
        if rotation_matrix.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3")
        R = rotation_matrix
        transformed_tensor = R @ self.tensor @ R.T
        return StrainTensor.from_array(transformed_tensor)
    def __add__(self, other: 'StrainTensor') -> 'StrainTensor':
        """Add strain tensors."""
        return StrainTensor.from_array(self.tensor + other.tensor)
    def __sub__(self, other: 'StrainTensor') -> 'StrainTensor':
        """Subtract strain tensors."""
        return StrainTensor.from_array(self.tensor - other.tensor)
    def __mul__(self, scalar: float) -> 'StrainTensor':
        """Multiply strain tensor by scalar."""
        return StrainTensor.from_array(scalar * self.tensor)
class IsotropicElasticity:
    """
    Isotropic linear elasticity relationships.
    Features:
    - Stress-strain constitutive relationships
    - Compliance and stiffness matrices
    - Elastic wave velocities
    - Energy calculations
    Examples:
        >>> elasticity = IsotropicElasticity(youngs_modulus=200e9, poissons_ratio=0.3)
        >>> stress = elasticity.stress_from_strain(strain_tensor)
        >>> strain = elasticity.strain_from_stress(stress_tensor)
    """
    def __init__(self, youngs_modulus: float, poissons_ratio: float, density: float = 7850):
        """
        Initialize isotropic elastic material.
        Parameters:
            youngs_modulus: Young's modulus (Pa)
            poissons_ratio: Poisson's ratio
            density: Material density (kg/m³)
        """
        self.constants = ElasticConstants(youngs_modulus, poissons_ratio)
        self.density = density
        # Pre-compute stiffness and compliance matrices
        self._stiffness_matrix = self._compute_stiffness_matrix()
        self._compliance_matrix = self._compute_compliance_matrix()
    def _compute_stiffness_matrix(self) -> np.ndarray:
        """Compute 6x6 stiffness matrix in Voigt notation."""
        E = self.constants.youngs_modulus
        nu = self.constants.poissons_ratio
        factor = E / ((1 + nu) * (1 - 2 * nu))
        C = np.zeros((6, 6))
        # Diagonal terms
        C[0, 0] = C[1, 1] = C[2, 2] = factor * (1 - nu)
        C[3, 3] = C[4, 4] = C[5, 5] = factor * (1 - 2 * nu) / 2
        # Off-diagonal terms
        off_diag = factor * nu
        C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = off_diag
        return C
    def _compute_compliance_matrix(self) -> np.ndarray:
        """Compute 6x6 compliance matrix in Voigt notation."""
        E = self.constants.youngs_modulus
        nu = self.constants.poissons_ratio
        G = self.constants.shear_modulus
        S = np.zeros((6, 6))
        # Diagonal terms
        S[0, 0] = S[1, 1] = S[2, 2] = 1 / E
        S[3, 3] = S[4, 4] = S[5, 5] = 1 / G
        # Off-diagonal terms
        S[0, 1] = S[0, 2] = S[1, 0] = S[1, 2] = S[2, 0] = S[2, 1] = -nu / E
        return S
    @property
    def stiffness_matrix(self) -> np.ndarray:
        """Get stiffness matrix."""
        return self._stiffness_matrix.copy()
    @property
    def compliance_matrix(self) -> np.ndarray:
        """Get compliance matrix."""
        return self._compliance_matrix.copy()
    def stress_from_strain(self, strain: StrainTensor) -> StressTensor:
        """Calculate stress from strain using Hooke's law."""
        strain_voigt = strain.to_voigt(engineering_strain=True)
        stress_voigt = self._stiffness_matrix @ strain_voigt
        return StressTensor.from_voigt(stress_voigt)
    def strain_from_stress(self, stress: StressTensor) -> StrainTensor:
        """Calculate strain from stress using compliance."""
        stress_voigt = stress.to_voigt()
        strain_voigt = self._compliance_matrix @ stress_voigt
        return StrainTensor.from_voigt(strain_voigt, engineering_strain=True)
    def elastic_wave_velocities(self) -> Tuple[float, float]:
        """
        Calculate elastic wave velocities.
        Returns:
            Tuple of (longitudinal_velocity, transverse_velocity) in m/s
        """
        rho = self.density
        lam = self.constants.lame_first
        mu = self.constants.lame_second
        # Longitudinal wave velocity
        v_p = np.sqrt((lam + 2 * mu) / rho)
        # Transverse (shear) wave velocity
        v_s = np.sqrt(mu / rho)
        return v_p, v_s
    def elastic_energy_density(self, stress: StressTensor, strain: StrainTensor) -> float:
        """Calculate elastic energy density."""
        stress_voigt = stress.to_voigt()
        strain_voigt = strain.to_voigt(engineering_strain=True)
        return 0.5 * np.dot(stress_voigt, strain_voigt)
    def bulk_modulus_from_stress(self, hydrostatic_stress: float) -> float:
        """Calculate bulk modulus from hydrostatic stress state."""
        return self.constants.bulk_modulus
    def shear_modulus_from_stress(self, shear_stress: float, shear_strain: float) -> float:
        """Calculate shear modulus from pure shear state."""
        if abs(shear_strain) < 1e-12:
            warnings.warn("Shear strain is very small, calculation may be inaccurate")
            return self.constants.shear_modulus
        return shear_stress / shear_strain
class AnisotropicElasticity:
    """
    Anisotropic linear elasticity for general materials.
    Features:
    - General 6x6 stiffness matrix handling
    - Orthotropic and transversely isotropic materials
    - Coordinate transformations
    - Engineering constants calculation
    Examples:
        >>> # Orthotropic material
        >>> stiffness = create_orthotropic_stiffness(Ex, Ey, Ez, Gxy, Gxz, Gyz, nuxy, nuxz, nuyz)
        >>> elasticity = AnisotropicElasticity(stiffness)
        >>> stress = elasticity.stress_from_strain(strain)
    """
    def __init__(self, stiffness_matrix: np.ndarray, density: float = 7850):
        """
        Initialize anisotropic elastic material.
        Parameters:
            stiffness_matrix: 6x6 stiffness matrix in Voigt notation
            density: Material density (kg/m³)
        """
        if stiffness_matrix.shape != (6, 6):
            raise ValueError("Stiffness matrix must be 6x6")
        # Check symmetry
        if not np.allclose(stiffness_matrix, stiffness_matrix.T):
            warnings.warn("Stiffness matrix is not symmetric")
        # Check positive definiteness
        eigenvals = np.linalg.eigvals(stiffness_matrix)
        if np.any(eigenvals <= 0):
            warnings.warn("Stiffness matrix is not positive definite")
        self._stiffness_matrix = stiffness_matrix.copy()
        self._compliance_matrix = np.linalg.inv(stiffness_matrix)
        self.density = density
    @classmethod
    def orthotropic(cls, Ex: float, Ey: float, Ez: float,
                   Gxy: float, Gxz: float, Gyz: float,
                   nuxy: float, nuxz: float, nuyz: float,
                   density: float = 7850) -> 'AnisotropicElasticity':
        """
        Create orthotropic material from engineering constants.
        Parameters:
            Ex, Ey, Ez: Young's moduli in x, y, z directions
            Gxy, Gxz, Gyz: Shear moduli in xy, xz, yz planes
            nuxy, nuxz, nuyz: Poisson's ratios
            density: Material density
        """
        # Check reciprocal relations
        nuyx = nuxy * Ey / Ex
        nuzx = nuxz * Ez / Ex
        nuzy = nuyz * Ez / Ey
        # Compute compliance matrix
        S = np.zeros((6, 6))
        S[0, 0] = 1 / Ex
        S[1, 1] = 1 / Ey
        S[2, 2] = 1 / Ez
        S[3, 3] = 1 / Gyz
        S[4, 4] = 1 / Gxz
        S[5, 5] = 1 / Gxy
        S[0, 1] = S[1, 0] = -nuxy / Ex
        S[0, 2] = S[2, 0] = -nuxz / Ex
        S[1, 2] = S[2, 1] = -nuyz / Ey
        # Invert to get stiffness matrix
        C = np.linalg.inv(S)
        return cls(C, density)
    @classmethod
    def transversely_isotropic(cls, E1: float, E2: float, G12: float, G23: float,
                              nu12: float, nu23: float, density: float = 7850) -> 'AnisotropicElasticity':
        """
        Create transversely isotropic material.
        Parameters:
            E1: Young's modulus in fiber direction
            E2: Young's modulus transverse to fiber
            G12: In-plane shear modulus
            G23: Out-of-plane shear modulus
            nu12: In-plane Poisson's ratio
            nu23: Out-of-plane Poisson's ratio
            density: Material density
        """
        return cls.orthotropic(
            Ex=E1, Ey=E2, Ez=E2,
            Gxy=G12, Gxz=G12, Gyz=G23,
            nuxy=nu12, nuxz=nu12, nuyz=nu23,
            density=density
        )
    @property
    def stiffness_matrix(self) -> np.ndarray:
        """Get stiffness matrix."""
        return self._stiffness_matrix.copy()
    @property
    def compliance_matrix(self) -> np.ndarray:
        """Get compliance matrix."""
        return self._compliance_matrix.copy()
    def stress_from_strain(self, strain: StrainTensor) -> StressTensor:
        """Calculate stress from strain."""
        strain_voigt = strain.to_voigt(engineering_strain=True)
        stress_voigt = self._stiffness_matrix @ strain_voigt
        return StressTensor.from_voigt(stress_voigt)
    def strain_from_stress(self, stress: StressTensor) -> StrainTensor:
        """Calculate strain from stress."""
        stress_voigt = stress.to_voigt()
        strain_voigt = self._compliance_matrix @ stress_voigt
        return StrainTensor.from_voigt(strain_voigt, engineering_strain=True)
    def transform_stiffness(self, rotation_matrix: np.ndarray) -> np.ndarray:
        """Transform stiffness matrix to new coordinate system."""
        if rotation_matrix.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3")
        # Create 6x6 transformation matrix for Voigt notation
        T = self._create_voigt_transformation_matrix(rotation_matrix)
        # Transform stiffness matrix
        C_transformed = T @ self._stiffness_matrix @ T.T
        return C_transformed
    def _create_voigt_transformation_matrix(self, R: np.ndarray) -> np.ndarray:
        """Create 6x6 transformation matrix for Voigt notation."""
        T = np.zeros((6, 6))
        # Direct terms
        for i in range(3):
            for j in range(3):
                T[i, j] = R[i, j]**2
        # Shear terms
        T[0, 3] = 2 * R[0, 1] * R[0, 2]
        T[0, 4] = 2 * R[0, 0] * R[0, 2]
        T[0, 5] = 2 * R[0, 0] * R[0, 1]
        T[1, 3] = 2 * R[1, 1] * R[1, 2]
        T[1, 4] = 2 * R[1, 0] * R[1, 2]
        T[1, 5] = 2 * R[1, 0] * R[1, 1]
        T[2, 3] = 2 * R[2, 1] * R[2, 2]
        T[2, 4] = 2 * R[2, 0] * R[2, 2]
        T[2, 5] = 2 * R[2, 0] * R[2, 1]
        # Mixed terms
        T[3, 0] = R[1, 0] * R[2, 0]
        T[3, 1] = R[1, 1] * R[2, 1]
        T[3, 2] = R[1, 2] * R[2, 2]
        T[3, 3] = R[1, 1] * R[2, 2] + R[1, 2] * R[2, 1]
        T[3, 4] = R[1, 0] * R[2, 2] + R[1, 2] * R[2, 0]
        T[3, 5] = R[1, 0] * R[2, 1] + R[1, 1] * R[2, 0]
        T[4, 0] = R[0, 0] * R[2, 0]
        T[4, 1] = R[0, 1] * R[2, 1]
        T[4, 2] = R[0, 2] * R[2, 2]
        T[4, 3] = R[0, 1] * R[2, 2] + R[0, 2] * R[2, 1]
        T[4, 4] = R[0, 0] * R[2, 2] + R[0, 2] * R[2, 0]
        T[4, 5] = R[0, 0] * R[2, 1] + R[0, 1] * R[2, 0]
        T[5, 0] = R[0, 0] * R[1, 0]
        T[5, 1] = R[0, 1] * R[1, 1]
        T[5, 2] = R[0, 2] * R[1, 2]
        T[5, 3] = R[0, 1] * R[1, 2] + R[0, 2] * R[1, 1]
        T[5, 4] = R[0, 0] * R[1, 2] + R[0, 2] * R[1, 0]
        T[5, 5] = R[0, 0] * R[1, 1] + R[0, 1] * R[1, 0]
        return T
    def elastic_energy_density(self, stress: StressTensor, strain: StrainTensor) -> float:
        """Calculate elastic energy density."""
        stress_voigt = stress.to_voigt()
        strain_voigt = strain.to_voigt(engineering_strain=True)
        return 0.5 * np.dot(stress_voigt, strain_voigt)
def create_material_database() -> dict:
    """Create database of common engineering materials."""
    return {
        'steel': ElasticConstants(youngs_modulus=200e9, poissons_ratio=0.3),
        'aluminum': ElasticConstants(youngs_modulus=70e9, poissons_ratio=0.33),
        'copper': ElasticConstants(youngs_modulus=110e9, poissons_ratio=0.35),
        'titanium': ElasticConstants(youngs_modulus=116e9, poissons_ratio=0.32),
        'concrete': ElasticConstants(youngs_modulus=30e9, poissons_ratio=0.2),
        'glass': ElasticConstants(youngs_modulus=70e9, poissons_ratio=0.22),
        'rubber': ElasticConstants(youngs_modulus=1e6, poissons_ratio=0.49),
        'carbon_fiber': ElasticConstants(youngs_modulus=230e9, poissons_ratio=0.2)
    }