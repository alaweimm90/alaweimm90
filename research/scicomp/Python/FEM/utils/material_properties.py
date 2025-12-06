"""
Material Properties for Finite Element Analysis
Comprehensive material property definitions and constitutive models
for various engineering materials and behaviors.
"""
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
@dataclass
class MaterialProperty:
    """
    Base material property definition.
    All materials must define basic mechanical properties.
    """
    name: str
    youngs_modulus: float  # Pa
    poissons_ratio: float
    density: float  # kg/m³
    thermal_expansion: float = 0.0  # 1/K
    thermal_conductivity: float = 0.0  # W/(m·K)
    specific_heat: float = 0.0  # J/(kg·K)
    def __post_init__(self):
        """Validate material properties."""
        if self.youngs_modulus <= 0:
            raise ValueError("Young's modulus must be positive")
        if not (-1 < self.poissons_ratio < 0.5):
            raise ValueError("Poisson's ratio must be between -1 and 0.5")
        if self.density <= 0:
            raise ValueError("Density must be positive")
    @property
    def shear_modulus(self) -> float:
        """Calculate shear modulus."""
        return self.youngs_modulus / (2 * (1 + self.poissons_ratio))
    @property
    def bulk_modulus(self) -> float:
        """Calculate bulk modulus."""
        return self.youngs_modulus / (3 * (1 - 2 * self.poissons_ratio))
    @property
    def lame_first(self) -> float:
        """Calculate first Lamé parameter."""
        return (self.youngs_modulus * self.poissons_ratio) / \
               ((1 + self.poissons_ratio) * (1 - 2 * self.poissons_ratio))
    @property
    def lame_second(self) -> float:
        """Calculate second Lamé parameter (shear modulus)."""
        return self.shear_modulus
    def elastic_wave_velocities(self) -> Tuple[float, float]:
        """
        Calculate elastic wave velocities.
        Returns:
            Tuple of (P-wave velocity, S-wave velocity) in m/s
        """
        lambda_param = self.lame_first
        mu = self.shear_modulus
        rho = self.density
        v_p = np.sqrt((lambda_param + 2 * mu) / rho)
        v_s = np.sqrt(mu / rho)
        return v_p, v_s
class IsotropicMaterial(MaterialProperty):
    """Isotropic linear elastic material."""
    def stiffness_matrix_3d(self) -> np.ndarray:
        """
        Get 3D stiffness matrix in Voigt notation.
        Returns:
            6x6 stiffness matrix
        """
        E = self.youngs_modulus
        nu = self.poissons_ratio
        factor = E / ((1 + nu) * (1 - 2 * nu))
        C = np.zeros((6, 6))
        # Diagonal terms
        C[0, 0] = C[1, 1] = C[2, 2] = factor * (1 - nu)
        C[3, 3] = C[4, 4] = C[5, 5] = factor * (1 - 2 * nu) / 2
        # Off-diagonal terms
        off_diag = factor * nu
        C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = off_diag
        return C
    def compliance_matrix_3d(self) -> np.ndarray:
        """
        Get 3D compliance matrix in Voigt notation.
        Returns:
            6x6 compliance matrix
        """
        E = self.youngs_modulus
        nu = self.poissons_ratio
        G = self.shear_modulus
        S = np.zeros((6, 6))
        # Diagonal terms
        S[0, 0] = S[1, 1] = S[2, 2] = 1 / E
        S[3, 3] = S[4, 4] = S[5, 5] = 1 / G
        # Off-diagonal terms
        S[0, 1] = S[0, 2] = S[1, 0] = S[1, 2] = S[2, 0] = S[2, 1] = -nu / E
        return S
    def plane_stress_matrix(self) -> np.ndarray:
        """
        Get plane stress stiffness matrix.
        Returns:
            3x3 stiffness matrix for plane stress
        """
        E = self.youngs_modulus
        nu = self.poissons_ratio
        factor = E / (1 - nu**2)
        return factor * np.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1 - nu) / 2]
        ])
    def plane_strain_matrix(self) -> np.ndarray:
        """
        Get plane strain stiffness matrix.
        Returns:
            3x3 stiffness matrix for plane strain
        """
        E = self.youngs_modulus
        nu = self.poissons_ratio
        factor = E / ((1 + nu) * (1 - 2 * nu))
        return factor * np.array([
            [1 - nu, nu, 0],
            [nu, 1 - nu, 0],
            [0, 0, (1 - 2 * nu) / 2]
        ])
@dataclass
class OrthotropicMaterial(MaterialProperty):
    """
    Orthotropic material with different properties in three directions.
    Common for composite materials and wood.
    """
    youngs_modulus_y: float = 0.0  # Pa, y-direction
    youngs_modulus_z: float = 0.0  # Pa, z-direction
    poissons_ratio_xy: float = 0.0
    poissons_ratio_xz: float = 0.0
    poissons_ratio_yz: float = 0.0
    shear_modulus_xy: float = 0.0  # Pa
    shear_modulus_xz: float = 0.0  # Pa
    shear_modulus_yz: float = 0.0  # Pa
    def __post_init__(self):
        """Validate orthotropic material properties."""
        super().__post_init__()
        # Set default values if not provided
        if self.youngs_modulus_y == 0.0:
            self.youngs_modulus_y = self.youngs_modulus
        if self.youngs_modulus_z == 0.0:
            self.youngs_modulus_z = self.youngs_modulus
        if self.poissons_ratio_xy == 0.0:
            self.poissons_ratio_xy = self.poissons_ratio
        if self.poissons_ratio_xz == 0.0:
            self.poissons_ratio_xz = self.poissons_ratio
        if self.poissons_ratio_yz == 0.0:
            self.poissons_ratio_yz = self.poissons_ratio
        if self.shear_modulus_xy == 0.0:
            self.shear_modulus_xy = self.shear_modulus
        if self.shear_modulus_xz == 0.0:
            self.shear_modulus_xz = self.shear_modulus
        if self.shear_modulus_yz == 0.0:
            self.shear_modulus_yz = self.shear_modulus
    def stiffness_matrix_3d(self) -> np.ndarray:
        """
        Get 3D orthotropic stiffness matrix.
        Returns:
            6x6 stiffness matrix
        """
        Ex, Ey, Ez = self.youngs_modulus, self.youngs_modulus_y, self.youngs_modulus_z
        vxy, vxz, vyz = self.poissons_ratio_xy, self.poissons_ratio_xz, self.poissons_ratio_yz
        Gxy, Gxz, Gyz = self.shear_modulus_xy, self.shear_modulus_xz, self.shear_modulus_yz
        # Calculate reciprocal Poisson's ratios
        vyx = vxy * Ey / Ex
        vzx = vxz * Ez / Ex
        vzy = vyz * Ez / Ey
        # Compliance matrix terms
        denominator = 1 - vxy * vyx - vxz * vzx - vyz * vzy - 2 * vxy * vyz * vzx
        C = np.zeros((6, 6))
        # Diagonal terms
        C[0, 0] = Ex * (1 - vyz * vzy) / denominator
        C[1, 1] = Ey * (1 - vxz * vzx) / denominator
        C[2, 2] = Ez * (1 - vxy * vyx) / denominator
        C[3, 3] = Gyz
        C[4, 4] = Gxz
        C[5, 5] = Gxy
        # Off-diagonal terms
        C[0, 1] = C[1, 0] = Ex * (vyx + vzx * vyz) / denominator
        C[0, 2] = C[2, 0] = Ex * (vzx + vyx * vyz) / denominator
        C[1, 2] = C[2, 1] = Ey * (vzy + vzx * vxy) / denominator
        return C
@dataclass
class ViscoelasticMaterial(MaterialProperty):
    """
    Linear viscoelastic material with time-dependent properties.
    Uses Prony series representation.
    """
    prony_times: List[float] = field(default_factory=list)  # Relaxation times
    prony_moduli: List[float] = field(default_factory=list)  # Relaxation moduli
    long_term_modulus: float = 0.0  # Long-term modulus
    def __post_init__(self):
        """Validate viscoelastic properties."""
        super().__post_init__()
        if len(self.prony_times) != len(self.prony_moduli):
            raise ValueError("Prony times and moduli must have same length")
        if self.long_term_modulus == 0.0:
            self.long_term_modulus = self.youngs_modulus * 0.1  # Default 10% long-term
    def relaxation_modulus(self, time: float) -> float:
        """
        Calculate relaxation modulus at given time.
        Parameters:
            time: Time (s)
        Returns:
            Relaxation modulus
        """
        E_t = self.long_term_modulus
        for tau, E_i in zip(self.prony_times, self.prony_moduli):
            E_t += E_i * np.exp(-time / tau)
        return E_t
class MaterialLibrary:
    """
    Library of common engineering materials.
    Provides standard material properties for various materials.
    """
    @staticmethod
    def steel_mild() -> IsotropicMaterial:
        """Mild steel properties."""
        return IsotropicMaterial(
            name="Mild Steel",
            youngs_modulus=200e9,  # Pa
            poissons_ratio=0.30,
            density=7850,  # kg/m³
            thermal_expansion=12e-6,  # 1/K
            thermal_conductivity=50,  # W/(m·K)
            specific_heat=460  # J/(kg·K)
        )
    @staticmethod
    def steel_stainless() -> IsotropicMaterial:
        """Stainless steel properties."""
        return IsotropicMaterial(
            name="Stainless Steel",
            youngs_modulus=193e9,
            poissons_ratio=0.27,
            density=8000,
            thermal_expansion=17e-6,
            thermal_conductivity=16,
            specific_heat=500
        )
    @staticmethod
    def aluminum_6061() -> IsotropicMaterial:
        """Aluminum 6061-T6 properties."""
        return IsotropicMaterial(
            name="Aluminum 6061-T6",
            youngs_modulus=68.9e9,
            poissons_ratio=0.33,
            density=2700,
            thermal_expansion=23.6e-6,
            thermal_conductivity=167,
            specific_heat=896
        )
    @staticmethod
    def titanium_ti6al4v() -> IsotropicMaterial:
        """Titanium Ti-6Al-4V properties."""
        return IsotropicMaterial(
            name="Titanium Ti-6Al-4V",
            youngs_modulus=113.8e9,
            poissons_ratio=0.342,
            density=4430,
            thermal_expansion=8.6e-6,
            thermal_conductivity=6.7,
            specific_heat=563
        )
    @staticmethod
    def concrete() -> IsotropicMaterial:
        """Concrete properties."""
        return IsotropicMaterial(
            name="Concrete",
            youngs_modulus=30e9,
            poissons_ratio=0.20,
            density=2400,
            thermal_expansion=10e-6,
            thermal_conductivity=1.7,
            specific_heat=880
        )
    @staticmethod
    def wood_pine() -> OrthotropicMaterial:
        """Pine wood properties (orthotropic)."""
        return OrthotropicMaterial(
            name="Pine Wood",
            youngs_modulus=12e9,  # Longitudinal (grain direction)
            youngs_modulus_y=0.8e9,  # Radial
            youngs_modulus_z=0.8e9,  # Tangential
            poissons_ratio=0.37,  # xy
            poissons_ratio_xy=0.37,
            poissons_ratio_xz=0.44,
            poissons_ratio_yz=0.54,
            density=500,
            shear_modulus_xy=0.7e9,
            shear_modulus_xz=0.7e9,
            shear_modulus_yz=0.05e9,
            thermal_expansion=4e-6,
            thermal_conductivity=0.12,
            specific_heat=1380
        )
    @staticmethod
    def carbon_fiber_epoxy() -> OrthotropicMaterial:
        """Carbon fiber/epoxy composite properties."""
        return OrthotropicMaterial(
            name="Carbon Fiber/Epoxy",
            youngs_modulus=150e9,  # Fiber direction
            youngs_modulus_y=10e9,  # Transverse
            youngs_modulus_z=10e9,  # Through thickness
            poissons_ratio=0.30,
            poissons_ratio_xy=0.30,
            poissons_ratio_xz=0.30,
            poissons_ratio_yz=0.40,
            density=1600,
            shear_modulus_xy=5e9,
            shear_modulus_xz=5e9,
            shear_modulus_yz=3.5e9,
            thermal_expansion=1e-6,  # Fiber direction
            thermal_conductivity=1.0,
            specific_heat=1050
        )
    @staticmethod
    def rubber_natural() -> ViscoelasticMaterial:
        """Natural rubber with viscoelastic properties."""
        return ViscoelasticMaterial(
            name="Natural Rubber",
            youngs_modulus=1.5e6,
            poissons_ratio=0.49,
            density=920,
            prony_times=[0.1, 1.0, 10.0, 100.0],  # seconds
            prony_moduli=[0.3e6, 0.2e6, 0.15e6, 0.1e6],  # Pa
            long_term_modulus=0.5e6,
            thermal_expansion=200e-6,
            thermal_conductivity=0.16,
            specific_heat=1900
        )
    @staticmethod
    def glass() -> IsotropicMaterial:
        """Soda-lime glass properties."""
        return IsotropicMaterial(
            name="Soda-Lime Glass",
            youngs_modulus=70e9,
            poissons_ratio=0.22,
            density=2500,
            thermal_expansion=9e-6,
            thermal_conductivity=1.0,
            specific_heat=840
        )
    @staticmethod
    def copper() -> IsotropicMaterial:
        """Copper properties."""
        return IsotropicMaterial(
            name="Copper",
            youngs_modulus=117e9,
            poissons_ratio=0.35,
            density=8960,
            thermal_expansion=17e-6,
            thermal_conductivity=400,
            specific_heat=385
        )
    @staticmethod
    def list_materials() -> List[str]:
        """List all available materials."""
        return [
            'steel_mild', 'steel_stainless', 'aluminum_6061', 'titanium_ti6al4v',
            'concrete', 'wood_pine', 'carbon_fiber_epoxy', 'rubber_natural',
            'glass', 'copper'
        ]
    @classmethod
    def get_material(cls, material_name: str) -> MaterialProperty:
        """
        Get material by name.
        Parameters:
            material_name: Name of material
        Returns:
            Material property object
        """
        method_name = material_name.lower()
        if hasattr(cls, method_name):
            return getattr(cls, method_name)()
        else:
            available = cls.list_materials()
            raise ValueError(f"Material '{material_name}' not found. "
                           f"Available materials: {available}")
def create_custom_material(name: str, youngs_modulus: float, poissons_ratio: float,
                          density: float, **kwargs) -> IsotropicMaterial:
    """
    Create custom isotropic material.
    Parameters:
        name: Material name
        youngs_modulus: Young's modulus (Pa)
        poissons_ratio: Poisson's ratio
        density: Density (kg/m³)
        **kwargs: Additional material properties
    Returns:
        Custom material object
    """
    return IsotropicMaterial(
        name=name,
        youngs_modulus=youngs_modulus,
        poissons_ratio=poissons_ratio,
        density=density,
        **kwargs
    )
def scale_material_properties(material: MaterialProperty, scale_factor: float) -> MaterialProperty:
    """
    Scale material properties for model scaling.
    Parameters:
        material: Original material
        scale_factor: Geometric scale factor
    Returns:
        Scaled material
    """
    # Create new material with scaled properties
    scaled_material = type(material)(
        name=f"{material.name} (Scaled {scale_factor}x)",
        youngs_modulus=material.youngs_modulus,  # Modulus unchanged
        poissons_ratio=material.poissons_ratio,  # Poisson's ratio unchanged
        density=material.density,  # Density unchanged
        thermal_expansion=material.thermal_expansion,
        thermal_conductivity=material.thermal_conductivity / scale_factor,
        specific_heat=material.specific_heat
    )
    return scaled_material