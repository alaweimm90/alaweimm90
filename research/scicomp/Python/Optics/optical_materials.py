"""Optical Materials Module.
This module provides comprehensive optical material properties including
refractive index models, dispersion analysis, and material databases.
Author: Berkeley SciComp Team
Date: 2024
"""
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Union, Dict, List, Tuple, Optional, Callable
from abc import ABC, abstractmethod
import json
import warnings
@dataclass
class MaterialProperties:
    """Container for optical material properties."""
    name: str
    refractive_index: Union[float, Callable]
    dispersion_model: str
    parameters: Dict
    wavelength_range: Tuple[float, float]  # meters
    transmission_range: Tuple[float, float] = (0.0, 1.0)  # fractional
    absorption_coefficient: Optional[Callable] = None
    thermal_dn_dt: float = 0.0  # dn/dT (1/K)
    nonlinear_index: float = 0.0  # n2 (m²/W)
class DispersionModel(ABC):
    """Abstract base class for dispersion models."""
    @abstractmethod
    def refractive_index(self, wavelength: float, temperature: float = 293.15) -> float:
        """Calculate refractive index at given wavelength and temperature.
        Args:
            wavelength: Wavelength in meters
            temperature: Temperature in Kelvin
        Returns:
            Refractive index
        """
        pass
    @abstractmethod
    def group_index(self, wavelength: float) -> float:
        """Calculate group index.
        Args:
            wavelength: Wavelength in meters
        Returns:
            Group index
        """
        pass
class Sellmeier(DispersionModel):
    """Sellmeier dispersion model."""
    def __init__(self, B1: float, B2: float, B3: float,
                 C1: float, C2: float, C3: float):
        """Initialize Sellmeier model.
        Args:
            B1, B2, B3: Sellmeier B coefficients
            C1, C2, C3: Sellmeier C coefficients (μm²)
        """
        self.B1, self.B2, self.B3 = B1, B2, B3
        self.C1, self.C2, self.C3 = C1, C2, C3
    def refractive_index(self, wavelength: float, temperature: float = 293.15) -> float:
        """Calculate refractive index using Sellmeier equation.
        n²(λ) = 1 + B₁λ²/(λ² - C₁) + B₂λ²/(λ² - C₂) + B₃λ²/(λ² - C₃)
        """
        lambda_um = wavelength * 1e6  # Convert to micrometers
        lambda_sq = lambda_um**2
        n_squared = (1 +
                    self.B1 * lambda_sq / (lambda_sq - self.C1) +
                    self.B2 * lambda_sq / (lambda_sq - self.C2) +
                    self.B3 * lambda_sq / (lambda_sq - self.C3))
        return np.sqrt(n_squared)
    def group_index(self, wavelength: float) -> float:
        """Calculate group index ng = n - λ(dn/dλ)."""
        # Numerical derivative
        dlambda = wavelength * 1e-6
        n_plus = self.refractive_index(wavelength + dlambda)
        n_minus = self.refractive_index(wavelength - dlambda)
        dn_dlambda = (n_plus - n_minus) / (2 * dlambda)
        n = self.refractive_index(wavelength)
        return n - wavelength * dn_dlambda
class Cauchy(DispersionModel):
    """Cauchy dispersion model."""
    def __init__(self, A: float, B: float = 0, C: float = 0):
        """Initialize Cauchy model.
        Args:
            A, B, C: Cauchy coefficients
        """
        self.A, self.B, self.C = A, B, C
    def refractive_index(self, wavelength: float, temperature: float = 293.15) -> float:
        """Calculate refractive index using Cauchy equation.
        n(λ) = A + B/λ² + C/λ⁴
        """
        lambda_um = wavelength * 1e6  # Convert to micrometers
        return self.A + self.B / lambda_um**2 + self.C / lambda_um**4
    def group_index(self, wavelength: float) -> float:
        """Calculate group index."""
        lambda_um = wavelength * 1e6
        # Analytical derivative
        dn_dlambda_um = -2 * self.B / lambda_um**3 - 4 * self.C / lambda_um**5
        dn_dlambda = dn_dlambda_um * 1e6  # Convert back to per meter
        n = self.refractive_index(wavelength)
        return n - wavelength * dn_dlambda
class LorentzDrude(DispersionModel):
    """Lorentz-Drude dispersion model for metals."""
    def __init__(self, epsilon_inf: float, oscillators: List[Dict]):
        """Initialize Lorentz-Drude model.
        Args:
            epsilon_inf: High-frequency dielectric constant
            oscillators: List of oscillator parameters
        """
        self.epsilon_inf = epsilon_inf
        self.oscillators = oscillators
    def dielectric_function(self, wavelength: float) -> complex:
        """Calculate complex dielectric function."""
        omega = 2 * np.pi * 2.99792458e8 / wavelength  # Angular frequency
        epsilon = self.epsilon_inf
        for osc in self.oscillators:
            omega_p = osc.get('plasma_frequency', 0)  # rad/s
            omega_0 = osc.get('resonance_frequency', 0)  # rad/s
            gamma = osc.get('damping', 0)  # rad/s
            epsilon += (omega_p**2 /
                       (omega_0**2 - omega**2 - 1j * gamma * omega))
        return epsilon
    def refractive_index(self, wavelength: float, temperature: float = 293.15) -> complex:
        """Calculate complex refractive index."""
        epsilon = self.dielectric_function(wavelength)
        return np.sqrt(epsilon)
    def group_index(self, wavelength: float) -> float:
        """Calculate group index (real part only)."""
        n = self.refractive_index(wavelength)
        return np.real(n)  # Simplified for absorbing materials
class OpticalMaterial:
    """Comprehensive optical material class."""
    def __init__(self, name: str, dispersion_model: DispersionModel,
                 wavelength_range: Tuple[float, float] = (400e-9, 1600e-9),
                 properties: Dict = None):
        """Initialize optical material.
        Args:
            name: Material name
            dispersion_model: Dispersion model object
            wavelength_range: Valid wavelength range (meters)
            properties: Additional material properties
        """
        self.name = name
        self.dispersion_model = dispersion_model
        self.wavelength_range = wavelength_range
        self.properties = properties or {}
    def refractive_index(self, wavelength: Union[float, np.ndarray],
                        temperature: float = 293.15) -> Union[float, np.ndarray]:
        """Get refractive index at wavelength(s)."""
        wavelength = np.asarray(wavelength)
        return self.dispersion_model.refractive_index(wavelength, temperature)
    def group_index(self, wavelength: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Get group index at wavelength(s)."""
        wavelength = np.asarray(wavelength)
        return self.dispersion_model.group_index(wavelength)
    def group_velocity_dispersion(self, wavelength: float) -> float:
        """Calculate group velocity dispersion (GVD).
        Args:
            wavelength: Wavelength (meters)
        Returns:
            GVD in ps²/km
        """
        c = 2.99792458e8  # m/s
        # Numerical second derivative
        dlambda = wavelength * 1e-6
        ng_plus = self.group_index(wavelength + dlambda)
        ng_center = self.group_index(wavelength)
        ng_minus = self.group_index(wavelength - dlambda)
        d2ng_dlambda2 = (ng_plus - 2*ng_center + ng_minus) / dlambda**2
        # Convert to standard units
        lambda_um = wavelength * 1e6
        gvd = -(lambda_um**3 / (2 * np.pi * c)) * d2ng_dlambda2 * 1e21  # ps²/km
        return gvd
    def chromatic_dispersion(self, wavelength: float) -> float:
        """Calculate chromatic dispersion parameter D.
        Args:
            wavelength: Wavelength (meters)
        Returns:
            Dispersion parameter in ps/(nm⋅km)
        """
        gvd = self.group_velocity_dispersion(wavelength)
        lambda_nm = wavelength * 1e9
        return -2 * np.pi * 2.99792458e8 * gvd / lambda_nm**2 * 1e-6
    def plot_dispersion(self, wavelength_range: Tuple[float, float] = None,
                       num_points: int = 1000):
        """Plot dispersion curve."""
        if wavelength_range is None:
            wavelength_range = self.wavelength_range
        wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], num_points)
        wavelengths_nm = wavelengths * 1e9
        n = self.refractive_index(wavelengths)
        ng = self.group_index(wavelengths)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        # Refractive index
        ax1.plot(wavelengths_nm, n, color='#003262', linewidth=2, label='n')
        ax1.plot(wavelengths_nm, ng, color='#FDB515', linewidth=2, label='ng')
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('Refractive Index')
        ax1.set_title(f'{self.name} - Refractive Index')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # Group velocity dispersion
        gvd = [self.group_velocity_dispersion(w) for w in wavelengths]
        ax2.plot(wavelengths_nm, gvd, color='#003262', linewidth=2)
        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel('GVD (ps²/km)')
        ax2.set_title(f'{self.name} - Group Velocity Dispersion')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
# Material database
def create_material_database() -> Dict[str, OpticalMaterial]:
    """Create database of common optical materials."""
    materials = {}
    # BK7 Glass (Schott)
    bk7_sellmeier = Sellmeier(
        B1=1.03961212, B2=0.231792344, B3=1.01046945,
        C1=6.00069867e-3, C2=2.00179144e-2, C3=103.560653
    )
    materials['BK7'] = OpticalMaterial(
        'BK7', bk7_sellmeier, (310e-9, 2500e-9),
        {'abbe_number': 64.17, 'density': 2.51, 'thermal_expansion': 7.1e-6}
    )
    # Fused Silica
    silica_sellmeier = Sellmeier(
        B1=0.6961663, B2=0.4079426, B3=0.8974794,
        C1=4.67914826e-3, C2=1.35120631e-2, C3=97.9340025
    )
    materials['SiO2'] = OpticalMaterial(
        'Fused Silica', silica_sellmeier, (210e-9, 3700e-9),
        {'abbe_number': 67.8, 'density': 2.20, 'thermal_expansion': 0.5e-6}
    )
    # Silicon
    si_sellmeier = Sellmeier(
        B1=10.6684293, B2=0.0030434748, B3=1.54133408,
        C1=0.301516485, C2=1.13475115, C3=1104.0
    )
    materials['Si'] = OpticalMaterial(
        'Silicon', si_sellmeier, (1200e-9, 14000e-9),
        {'bandgap': 1.12, 'density': 2.33, 'thermal_expansion': 2.6e-6}
    )
    # Sapphire (Al2O3)
    sapphire_sellmeier = Sellmeier(
        B1=1.4313493, B2=0.65054713, B3=5.3414021,
        C1=5.2799261e-3, C2=1.42382647e-2, C3=325.017834
    )
    materials['Al2O3'] = OpticalMaterial(
        'Sapphire', sapphire_sellmeier, (150e-9, 5500e-9),
        {'abbe_number': 72.2, 'density': 3.98, 'thermal_expansion': 5.4e-6}
    )
    # Water
    water_sellmeier = Sellmeier(
        B1=5.684027565e-1, B2=1.726177391e-1, B3=2.086189578e-2,
        C1=5.101829712e-3, C2=1.821153936e-2, C3=2.620722293e-2
    )
    materials['H2O'] = OpticalMaterial(
        'Water', water_sellmeier, (200e-9, 200000e-9),
        {'density': 1.0, 'thermal_expansion': 214e-6}
    )
    # Air (at 15°C, 760 Torr, 0.03% CO2)
    air_cauchy = Cauchy(A=1.000293)
    materials['air'] = OpticalMaterial(
        'Air', air_cauchy, (200e-9, 2000000e-9),
        {'density': 1.225e-3, 'pressure_dependence': True}
    )
    return materials
# Utility functions
def calculate_refractive_index(material: Union[str, OpticalMaterial],
                             wavelength: float, temperature: float = 293.15) -> float:
    """Calculate refractive index for material.
    Args:
        material: Material name or OpticalMaterial object
        wavelength: Wavelength (meters)
        temperature: Temperature (Kelvin)
    Returns:
        Refractive index
    """
    if isinstance(material, str):
        # Load from database
        materials_db = create_material_database()
        if material in materials_db:
            material_obj = materials_db[material]
        else:
            # Default values for common materials
            defaults = {
                'air': 1.000293,
                'vacuum': 1.0,
                'water': 1.333,
                'glass': 1.5,
                'diamond': 2.4
            }
            return defaults.get(material.lower(), 1.5)
    else:
        material_obj = material
    return material_obj.refractive_index(wavelength, temperature)
def dispersion_analysis(material: OpticalMaterial,
                       wavelength_range: Tuple[float, float] = None) -> Dict[str, Any]:
    """Perform comprehensive dispersion analysis.
    Args:
        material: OpticalMaterial object
        wavelength_range: Wavelength range for analysis (meters)
    Returns:
        Dictionary with dispersion analysis results
    """
    if wavelength_range is None:
        wavelength_range = material.wavelength_range
    wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], 1000)
    # Calculate dispersion properties
    n = material.refractive_index(wavelengths)
    ng = material.group_index(wavelengths)
    # Find zero dispersion wavelength
    gvd = [material.group_velocity_dispersion(w) for w in wavelengths]
    gvd = np.array(gvd)
    zero_disp_idx = np.where(np.diff(np.sign(gvd)))[0]
    zero_disp_wavelengths = wavelengths[zero_disp_idx] if len(zero_disp_idx) > 0 else []
    # Calculate Abbe number (if in visible range)
    lambda_d = 589.3e-9  # Sodium D-line
    lambda_f = 486.1e-9  # Hydrogen F-line
    lambda_c = 656.3e-9  # Hydrogen C-line
    abbe_number = None
    if (wavelength_range[0] <= lambda_c <= wavelength_range[1] and
        wavelength_range[0] <= lambda_f <= wavelength_range[1]):
        n_d = material.refractive_index(lambda_d)
        n_f = material.refractive_index(lambda_f)
        n_c = material.refractive_index(lambda_c)
        abbe_number = (n_d - 1) / (n_f - n_c)
    return {
        'wavelengths': wavelengths,
        'refractive_index': n,
        'group_index': ng,
        'gvd': gvd,
        'zero_dispersion_wavelengths': zero_disp_wavelengths,
        'abbe_number': abbe_number,
        'wavelength_range': wavelength_range
    }
def absorption_coefficient(material: str, wavelength: float) -> float:
    """Calculate absorption coefficient for material.
    Args:
        material: Material name
        wavelength: Wavelength (meters)
    Returns:
        Absorption coefficient (1/m)
    """
    # Simplified absorption models
    absorption_data = {
        'BK7': lambda w: 0.001,  # Very low absorption in visible
        'SiO2': lambda w: 0.0001 if w > 200e-9 else 100,  # UV cutoff
        'Si': lambda w: 1e6 if w < 1100e-9 else 0.01,  # Bandgap absorption
        'water': lambda w: 0.01 * (w * 1e6)**2,  # Increases with wavelength
        'air': lambda w: 1e-6  # Minimal absorption
    }
    if material in absorption_data:
        return absorption_data[material](wavelength)
    else:
        return 0.001  # Default low absorption
def group_velocity_dispersion(material: Union[str, OpticalMaterial],
                            wavelength: float) -> float:
    """Calculate group velocity dispersion.
    Args:
        material: Material name or OpticalMaterial object
        wavelength: Wavelength (meters)
    Returns:
        GVD in ps²/km
    """
    if isinstance(material, str):
        materials_db = create_material_database()
        if material in materials_db:
            material_obj = materials_db[material]
        else:
            raise ValueError(f"Unknown material: {material}")
    else:
        material_obj = material
    return material_obj.group_velocity_dispersion(wavelength)
# Demonstration
def demo_optical_materials():
    """Demonstrate optical materials functionality."""
    print("Optical Materials Demo")
    print("=====================")
    # Create material database
    materials_db = create_material_database()
    print(f"\nAvailable materials: {list(materials_db.keys())}")
    # Analyze BK7 glass
    bk7 = materials_db['BK7']
    print(f"\n1. BK7 Glass Analysis")
    print(f"Name: {bk7.name}")
    print(f"Wavelength range: {bk7.wavelength_range[0]*1e9:.0f}-{bk7.wavelength_range[1]*1e9:.0f} nm")
    # Calculate properties at specific wavelengths
    wavelengths = [486.1e-9, 589.3e-9, 656.3e-9]  # F, d, C lines
    for i, wl in enumerate(wavelengths):
        n = bk7.refractive_index(wl)
        ng = bk7.group_index(wl)
        gvd = bk7.group_velocity_dispersion(wl)
        line_names = ['F-line', 'd-line', 'C-line']
        print(f"{line_names[i]} ({wl*1e9:.1f} nm): n = {n:.6f}, ng = {ng:.6f}, GVD = {gvd:.2f} ps²/km")
    # Plot dispersion
    bk7.plot_dispersion()
    # Compare materials
    print("\n2. Material Comparison at 589.3 nm")
    wavelength = 589.3e-9
    for name, material in materials_db.items():
        if wavelength >= material.wavelength_range[0] and wavelength <= material.wavelength_range[1]:
            n = material.refractive_index(wavelength)
            print(f"{name:10s}: n = {n:.6f}")
    # Dispersion analysis
    print("\n3. Fused Silica Dispersion Analysis")
    silica = materials_db['SiO2']
    analysis = dispersion_analysis(silica, (400e-9, 1600e-9))
    if analysis['abbe_number']:
        print(f"Abbe number: {analysis['abbe_number']:.1f}")
    if len(analysis['zero_dispersion_wavelengths']) > 0:
        zdw = analysis['zero_dispersion_wavelengths'][0] * 1e9
        print(f"Zero dispersion wavelength: {zdw:.1f} nm")
if __name__ == "__main__":
    demo_optical_materials()