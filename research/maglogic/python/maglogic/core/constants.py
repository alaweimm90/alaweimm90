"""
Physical constants and material properties for micromagnetism.

SI units throughout. Material parameters from literature.

Author: Meshal Alawein
Email: meshal@berkeley.edu
License: MIT
"""

import numpy as np
from typing import Dict, Any

# Physical constants (SI units)
PHYSICAL_CONSTANTS: Dict[str, float] = {
    # Universal constants
    "mu_0": 4 * np.pi * 1e-7,           # Vacuum permeability (H/m)
    "epsilon_0": 8.854187817e-12,       # Vacuum permittivity (F/m)
    "c": 2.99792458e8,                  # Speed of light (m/s)
    "h": 6.62607015e-34,                # Planck constant (J⋅s)
    "hbar": 1.054571817e-34,            # Reduced Planck constant (J⋅s)
    "k_B": 1.380649e-23,                # Boltzmann constant (J/K)
    
    # Electron properties
    "e": 1.602176634e-19,               # Elementary charge (C)
    "m_e": 9.1093837015e-31,            # Electron mass (kg)
    "mu_B": 9.2740100783e-24,           # Bohr magneton (J/T)
    "g_e": 2.00231930436256,            # Electron g-factor
    
    # Derived magnetic constants
    "gamma_e": 1.7608592e11,            # Gyromagnetic ratio (rad⋅s⁻¹⋅T⁻¹)
    "alpha_fs": 7.2973525693e-3,        # Fine structure constant
}

# Material properties
MATERIAL_CONSTANTS: Dict[str, Dict[str, Any]] = {
    "permalloy_ni80fe20": {
        "Ms": 860e3,                     # Saturation magnetization (A/m)
        "A_ex": 13e-12,                  # Exchange constant (J/m)
        "K1": 0,                         # First-order anisotropy (J/m³)
        "K2": 0,                         # Second-order anisotropy (J/m³)
        "lambda_s": -7e-6,               # Magnetostriction constant
        "alpha": 0.01,                   # Gilbert damping parameter
        "gamma": 2.211e5,                # Gyromagnetic ratio (m⋅A⁻¹⋅s⁻¹)
        "density": 8700,                 # Mass density (kg/m³)
        "description": "Ni80Fe20 permalloy - standard NML material",
        "references": ["IEEE Mag. Lett. 2019", "J. Appl. Phys. 2020"]
    },
    
    "cofeb": {
        "Ms": 1600e3,                    # Saturation magnetization (A/m)
        "A_ex": 20e-12,                  # Exchange constant (J/m)
        "K1": 1000,                      # Interfacial anisotropy (J/m³)
        "K2": 0,                         # Second-order anisotropy (J/m³)
        "lambda_s": 60e-6,               # Magnetostriction constant
        "alpha": 0.004,                  # Gilbert damping parameter  
        "gamma": 2.21e5,                 # Gyromagnetic ratio (m⋅A⁻¹⋅s⁻¹)
        "density": 7500,                 # Mass density (kg/m³)
        "description": "CoFeB - MTJ and STT applications",
        "references": ["Nat. Mater. 2010", "Phys. Rev. B 2012"]
    },
    
    "iron": {
        "Ms": 1710e3,                    # Saturation magnetization (A/m)
        "A_ex": 21e-12,                  # Exchange constant (J/m)
        "K1": 48e3,                      # First-order anisotropy (J/m³)
        "K2": -4.2e3,                    # Second-order anisotropy (J/m³)
        "lambda_s": -7e-6,               # Magnetostriction constant
        "alpha": 0.002,                  # Gilbert damping parameter
        "gamma": 2.21e5,                 # Gyromagnetic ratio (m⋅A⁻¹⋅s⁻¹)
        "density": 7874,                 # Mass density (kg/m³)
        "description": "Pure iron - reference material",
        "references": ["Handbook of Magnetism", "Phys. Rev. B 1995"]
    },
    
    "heusler_co2mnsi": {
        "Ms": 1000e3,                    # Saturation magnetization (A/m)
        "A_ex": 17e-12,                  # Exchange constant (J/m)
        "K1": 10e3,                      # First-order anisotropy (J/m³)
        "K2": 0,                         # Second-order anisotropy (J/m³)
        "lambda_s": 0,                   # Magnetostriction constant
        "alpha": 0.001,                  # Gilbert damping parameter
        "gamma": 2.21e5,                 # Gyromagnetic ratio (m⋅A⁻¹⋅s⁻¹)
        "density": 7200,                 # Mass density (kg/m³)
        "description": "Co2MnSi Heusler alloy - half-metallic",
        "references": ["Phys. Rev. B 2006", "Appl. Phys. Lett. 2008"]
    },
    
    "gadolinium": {
        "Ms": 2060e3,                    # Saturation magnetization (A/m)
        "A_ex": 8e-12,                   # Exchange constant (J/m)
        "K1": -600e3,                    # First-order anisotropy (J/m³)
        "K2": 0,                         # Second-order anisotropy (J/m³)
        "lambda_s": -1.5e-3,             # Magnetostriction constant
        "alpha": 0.1,                    # Gilbert damping parameter
        "gamma": 1.76e5,                 # Gyromagnetic ratio (m⋅A⁻¹⋅s⁻¹)
        "density": 7900,                 # Mass density (kg/m³)
        "description": "Gadolinium rare-earth ferromagnet",
        "references": ["Phys. Rev. 1960", "J. Magn. Magn. Mater. 1980"]
    }
}

# Default simulation parameters
SIMULATION_DEFAULTS: Dict[str, Any] = {
    # Spatial discretization
    "cell_size": 2e-9,                   # Default mesh cell size (m)
    "thickness": 10e-9,                  # Default film thickness (m)
    
    # Time integration
    "time_step": 1e-12,                  # Default time step (s)
    "final_time": 10e-9,                 # Default simulation time (s)
    "max_step_size": 1e-3,               # Maximum step size for adaptive algorithms
    
    # Convergence criteria
    "stopping_mxHxm": 0.01,              # Torque convergence criterion
    "stopping_dm_dt": 0.01,              # dm/dt stopping criterion (°/ns)
    
    # Field parameters
    "field_units": "A/m",                # Default field units
    "temperature": 0.0,                   # Default temperature (K)
    
    # Output parameters
    "data_table_step": 10e-12,           # Data table output interval (s)
    "magnetization_step": 100e-12,       # Magnetization output interval (s)
    
    # Geometry defaults
    "element_shape": "triangle",          # Default element shape
    "element_size": 100e-9,              # Default element size (m)
    "spacing": 30e-9,                    # Default inter-element spacing (m)
}

# Unit conversion factors
UNIT_CONVERSIONS: Dict[str, Dict[str, float]] = {
    "length": {
        "m": 1.0,
        "mm": 1e-3,
        "μm": 1e-6,
        "um": 1e-6,
        "nm": 1e-9,
        "Å": 1e-10,
        "angstrom": 1e-10,
    },
    
    "time": {
        "s": 1.0,
        "ms": 1e-3,
        "μs": 1e-6,
        "us": 1e-6,
        "ns": 1e-9,
        "ps": 1e-12,
        "fs": 1e-15,
    },
    
    "magnetic_field": {
        "A/m": 1.0,
        "Oe": 1000.0 / (4 * np.pi),      # Oersted to A/m
        "mT": 1000.0 / PHYSICAL_CONSTANTS["mu_0"],  # mT to A/m
        "T": 1.0 / PHYSICAL_CONSTANTS["mu_0"],      # T to A/m
        "kA/m": 1000.0,
        "MA/m": 1e6,
    },
    
    "magnetization": {
        "A/m": 1.0,
        "emu/cm³": 1000.0,               # CGS to SI
        "kA/m": 1000.0,
        "MA/m": 1e6,
    },
    
    "energy": {
        "J": 1.0,
        "eV": PHYSICAL_CONSTANTS["e"],
        "erg": 1e-7,
        "cal": 4.184,
        "meV": PHYSICAL_CONSTANTS["e"] * 1e-3,
    },
    
    "temperature": {
        "K": 1.0,
        "°C": lambda T: T + 273.15,      # Special case - need function
        "°F": lambda T: (T - 32) * 5/9 + 273.15,  # Special case
    }
}

# Shape anisotropy factors for common geometries
SHAPE_ANISOTROPY: Dict[str, Dict[str, float]] = {
    "sphere": {
        "Nx": 1/3, "Ny": 1/3, "Nz": 1/3,
        "description": "Isotropic sphere"
    },
    
    "thin_film": {
        "Nx": 0.0, "Ny": 0.0, "Nz": 1.0,
        "description": "Infinite thin film (thickness << lateral size)"
    },
    
    "wire_x": {
        "Nx": 0.0, "Ny": 0.5, "Nz": 0.5,
        "description": "Wire along x-direction"
    },
    
    "wire_y": {
        "Nx": 0.5, "Ny": 0.0, "Nz": 0.5,
        "description": "Wire along y-direction"
    },
    
    "wire_z": {
        "Nx": 0.5, "Ny": 0.5, "Nz": 0.0,
        "description": "Wire along z-direction"
    },
    
    "ellipsoid": {
        "description": "Ellipsoid - requires calculation based on aspect ratios"
    }
}

# Common temperature points for magnetic materials
TEMPERATURE_POINTS: Dict[str, float] = {
    "absolute_zero": 0.0,                # Absolute zero (K)
    "liquid_helium": 4.2,                # Liquid helium (K)
    "liquid_nitrogen": 77.0,             # Liquid nitrogen (K)
    "room_temperature": 293.15,          # Room temperature (K)
    "body_temperature": 310.15,          # Human body temperature (K)
    
    # Magnetic transition temperatures
    "iron_curie": 1043.0,                # Iron Curie temperature (K)
    "nickel_curie": 631.0,               # Nickel Curie temperature (K)
    "cobalt_curie": 1388.0,              # Cobalt Curie temperature (K)
    "gadolinium_curie": 293.0,           # Gadolinium Curie temperature (K)
}

# Validation ranges for simulation parameters
PARAMETER_RANGES: Dict[str, Dict[str, float]] = {
    "Ms": {"min": 1e3, "max": 3e6, "units": "A/m"},
    "A_ex": {"min": 1e-15, "max": 100e-12, "units": "J/m"},
    "K1": {"min": -1e6, "max": 1e6, "units": "J/m³"},
    "alpha": {"min": 1e-5, "max": 1.0, "units": "dimensionless"},
    "gamma": {"min": 1e4, "max": 3e5, "units": "m⋅A⁻¹⋅s⁻¹"},
    "temperature": {"min": 0.0, "max": 1000.0, "units": "K"},
    "cell_size": {"min": 0.1e-9, "max": 1e-6, "units": "m"},
    "time_step": {"min": 1e-18, "max": 1e-9, "units": "s"},
}

def get_material_parameter(material: str, parameter: str) -> float:
    """
    Get material parameter value.
    
    Args:
        material: Material name (e.g., 'permalloy_ni80fe20')
        parameter: Parameter name (e.g., 'Ms', 'A_ex')
    
    Returns:
        Parameter value in SI units
        
    Raises:
        KeyError: Unknown material or parameter
    """
    if material not in MATERIAL_CONSTANTS:
        available = list(MATERIAL_CONSTANTS.keys())
        raise KeyError(f"Material '{material}' not found. Available: {available}")
    
    if parameter not in MATERIAL_CONSTANTS[material]:
        available = list(MATERIAL_CONSTANTS[material].keys())
        raise KeyError(f"Parameter '{parameter}' not found for {material}. Available: {available}")
    
    return MATERIAL_CONSTANTS[material][parameter]

def list_materials() -> list:
    """Get list of available materials."""
    return list(MATERIAL_CONSTANTS.keys())

def get_material_info(material: str) -> Dict[str, Any]:
    """
    Get all parameters for a material.
    
    Args:
        material: Material name
        
    Returns:
        Dict with all material parameters
    """
    if material not in MATERIAL_CONSTANTS:
        available = list(MATERIAL_CONSTANTS.keys())
        raise KeyError(f"Material '{material}' not found. Available: {available}")
    
    return MATERIAL_CONSTANTS[material].copy()

def calculate_exchange_length(material: str) -> float:
    """
    Calculate exchange length for a material.
    
    Exchange length: l_ex = sqrt(2*A_ex / (mu_0 * Ms^2))
    
    Args:
        material: Material name
        
    Returns:
        Exchange length in meters
    """
    A_ex = get_material_parameter(material, "A_ex")
    Ms = get_material_parameter(material, "Ms")
    mu_0 = PHYSICAL_CONSTANTS["mu_0"]
    
    l_ex = np.sqrt(2 * A_ex / (mu_0 * Ms**2))
    return l_ex

def calculate_domain_wall_width(material: str) -> float:
    """
    Calculate domain wall width for a material.
    
    Domain wall width: δ_w = π * sqrt(A_ex / K_eff)
    For shape anisotropy: K_eff ≈ mu_0 * Ms^2 / 2
    
    Args:
        material: Material name
        
    Returns:
        Domain wall width in meters
    """
    A_ex = get_material_parameter(material, "A_ex")
    Ms = get_material_parameter(material, "Ms")
    mu_0 = PHYSICAL_CONSTANTS["mu_0"]
    
    # Use shape anisotropy as effective anisotropy
    K_eff = mu_0 * Ms**2 / 2
    
    delta_w = np.pi * np.sqrt(A_ex / K_eff)
    return delta_w

def thermal_energy(temperature: float) -> float:
    """
    Calculate thermal energy k_B * T.
    
    Args:
        temperature: Temperature in Kelvin
        
    Returns:
        Thermal energy in Joules
    """
    return PHYSICAL_CONSTANTS["k_B"] * temperature

def magnetic_energy_scale(material: str, volume: float) -> float:
    """
    Calculate magnetic energy scale for thermal activation.
    
    Energy scale: E_scale = mu_0 * Ms^2 * V / 2
    
    Args:
        material: Material name
        volume: Element volume in m³
        
    Returns:
        Energy scale in Joules
    """
    Ms = get_material_parameter(material, "Ms")
    mu_0 = PHYSICAL_CONSTANTS["mu_0"]
    
    E_scale = mu_0 * Ms**2 * volume / 2
    return E_scale

def validate_parameter(parameter: str, value: float) -> bool:
    """
    Validate if a parameter value is within reasonable physical limits.
    
    Args:
        parameter: Parameter name (e.g., 'Ms', 'A_ex', 'alpha')
        value: Parameter value to validate
        
    Returns:
        True if parameter is within valid range, False otherwise
    """
    if parameter not in PARAMETER_RANGES:
        return True  # No validation range defined
    
    param_range = PARAMETER_RANGES[parameter]
    return param_range["min"] <= value <= param_range["max"]