"""
MetaHub Scientific Computing Constants

Universal physical and mathematical constants for scientific computing.

Consolidated from SciComp modules:
- Elasticity/utils/constants.py
- FEM/utils/material_properties.py
- Multiple module constants

Usage:
    from metahub.libs.scientific_computing.core import constants

    G = constants.GRAVITATIONAL_CONSTANT
    c = constants.SPEED_OF_LIGHT
"""

import numpy as np

# Physical Constants (SI units)
PHYSICAL_CONSTANTS = {
    # Fundamental constants
    "speed_of_light": 299792458.0,  # m/s
    "gravitational_constant": 6.67430e-11,  # m^3 kg^-1 s^-2
    "planck_constant": 6.62607015e-34,  # J⋅s
    "boltzmann_constant": 1.380649e-23,  # J/K
    "avogadro_constant": 6.02214076e23,  # mol^-1
    "gas_constant": 8.314462618,  # J/(mol⋅K)

    # Electromagnetic
    "vacuum_permittivity": 8.8541878128e-12,  # F/m
    "vacuum_permeability": 1.25663706212e-6,  # H/m
    "elementary_charge": 1.602176634e-19,  # C

    # Atomic/Quantum
    "electron_mass": 9.1093837015e-31,  # kg
    "proton_mass": 1.67262192369e-27,  # kg
    "bohr_radius": 5.29177210903e-11,  # m
}

# Material Constants (common materials)
MATERIAL_CONSTANTS = {
    "steel": {
        "youngs_modulus": 200e9,  # Pa
        "poisson_ratio": 0.3,
        "density": 7850,  # kg/m^3
        "thermal_conductivity": 50.2,  # W/(m⋅K)
    },
    "aluminum": {
        "youngs_modulus": 69e9,  # Pa
        "poisson_ratio": 0.33,
        "density": 2700,  # kg/m^3
        "thermal_conductivity": 237,  # W/(m⋅K)
    },
    "copper": {
        "youngs_modulus": 130e9,  # Pa
        "poisson_ratio": 0.34,
        "density": 8960,  # kg/m^3
        "thermal_conductivity": 401,  # W/(m⋅K)
    },
}

# Mathematical Constants
MATH_CONSTANTS = {
    "pi": np.pi,
    "e": np.e,
    "golden_ratio": (1 + np.sqrt(5)) / 2,
}

# Convenience accessors
def get_physical_constant(name: str) -> float:
    """Get physical constant by name."""
    return PHYSICAL_CONSTANTS[name]

def get_material_property(material: str, property: str) -> float:
    """Get material property."""
    return MATERIAL_CONSTANTS[material][property]
