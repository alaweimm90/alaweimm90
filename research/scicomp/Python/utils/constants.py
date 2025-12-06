#!/usr/bin/env python3
"""
Physical Constants Module
Provides fundamental physical constants and unit conversion factors
used throughout scientific computing applications.
All constants are given in SI units unless otherwise specified.
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
import numpy as np
from typing import Dict, Union
# Fundamental constants (2018 CODATA values)
PHYSICAL_CONSTANTS: Dict[str, Dict[str, Union[float, str]]] = {
    # Universal constants
    'speed_of_light': {
        'value': 299792458.0,
        'unit': 'm/s',
        'uncertainty': 0.0,
        'symbol': 'c'
    },
    'planck_constant': {
        'value': 6.62607015e-34,
        'unit': 'J⋅s',
        'uncertainty': 0.0,
        'symbol': 'h'
    },
    'reduced_planck_constant': {
        'value': 1.0545718176461565e-34,
        'unit': 'J⋅s',
        'uncertainty': 0.0,
        'symbol': 'ℏ'
    },
    'elementary_charge': {
        'value': 1.602176634e-19,
        'unit': 'C',
        'uncertainty': 0.0,
        'symbol': 'e'
    },
    'boltzmann_constant': {
        'value': 1.380649e-23,
        'unit': 'J/K',
        'uncertainty': 0.0,
        'symbol': 'kB'
    },
    'avogadro_constant': {
        'value': 6.02214076e23,
        'unit': '1/mol',
        'uncertainty': 0.0,
        'symbol': 'NA'
    },
    # Electromagnetic constants
    'vacuum_permeability': {
        'value': 1.25663706212e-6,
        'unit': 'H/m',
        'uncertainty': 1.9e-16,
        'symbol': 'μ₀'
    },
    'vacuum_permittivity': {
        'value': 8.8541878128e-12,
        'unit': 'F/m',
        'uncertainty': 1.3e-21,
        'symbol': 'ε₀'
    },
    'fine_structure_constant': {
        'value': 7.2973525693e-3,
        'unit': 'dimensionless',
        'uncertainty': 1.1e-12,
        'symbol': 'α'
    },
    # Particle masses
    'electron_mass': {
        'value': 9.1093837015e-31,
        'unit': 'kg',
        'uncertainty': 2.8e-40,
        'symbol': 'me'
    },
    'proton_mass': {
        'value': 1.67262192369e-27,
        'unit': 'kg',
        'uncertainty': 5.1e-37,
        'symbol': 'mp'
    },
    'neutron_mass': {
        'value': 1.67492749804e-27,
        'unit': 'kg',
        'uncertainty': 9.5e-37,
        'symbol': 'mn'
    },
    'atomic_mass_unit': {
        'value': 1.66053906660e-27,
        'unit': 'kg',
        'uncertainty': 5.0e-37,
        'symbol': 'u'
    },
    # Atomic units (Hartree atomic units)
    'bohr_radius': {
        'value': 5.29177210903e-11,
        'unit': 'm',
        'uncertainty': 8.0e-21,
        'symbol': 'a₀'
    },
    'hartree_energy': {
        'value': 4.3597447222071e-18,
        'unit': 'J',
        'uncertainty': 8.5e-30,
        'symbol': 'Eh'
    },
    'rydberg_constant': {
        'value': 10973731.568160,
        'unit': '1/m',
        'uncertainty': 2.1e-05,
        'symbol': 'R∞'
    },
}
# Quick access to commonly used constants
h = PHYSICAL_CONSTANTS['planck_constant']['value']
hbar = PHYSICAL_CONSTANTS['reduced_planck_constant']['value']
c = PHYSICAL_CONSTANTS['speed_of_light']['value']
e = PHYSICAL_CONSTANTS['elementary_charge']['value']
me = PHYSICAL_CONSTANTS['electron_mass']['value']
mp = PHYSICAL_CONSTANTS['proton_mass']['value']
kb = PHYSICAL_CONSTANTS['boltzmann_constant']['value']
epsilon0 = PHYSICAL_CONSTANTS['vacuum_permittivity']['value']
mu0 = PHYSICAL_CONSTANTS['vacuum_permeability']['value']
NA = PHYSICAL_CONSTANTS['avogadro_constant']['value']
alpha = PHYSICAL_CONSTANTS['fine_structure_constant']['value']
a0 = PHYSICAL_CONSTANTS['bohr_radius']['value']
Eh = PHYSICAL_CONSTANTS['hartree_energy']['value']
# Common unit conversions
eV_to_J = e  # 1 eV = 1.602176634e-19 J
J_to_eV = 1.0 / eV_to_J
Ry_to_eV = Eh / (2 * eV_to_J)  # 1 Rydberg = 13.605693... eV
Ha_to_eV = Eh / eV_to_J  # 1 Hartree = 27.211386... eV
bohr_to_m = a0
m_to_bohr = 1.0 / a0
atomic_time_to_s = hbar / Eh  # Atomic unit of time
s_to_atomic_time = 1.0 / atomic_time_to_s
# Temperature conversions
def celsius_to_kelvin(T_celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return T_celsius + 273.15
def kelvin_to_celsius(T_kelvin: float) -> float:
    """Convert Kelvin to Celsius."""
    return T_kelvin - 273.15
def fahrenheit_to_kelvin(T_fahrenheit: float) -> float:
    """Convert Fahrenheit to Kelvin."""
    return (T_fahrenheit - 32.0) * 5.0/9.0 + 273.15
def kelvin_to_fahrenheit(T_kelvin: float) -> float:
    """Convert Kelvin to Fahrenheit."""
    return (T_kelvin - 273.15) * 9.0/5.0 + 32.0
# Energy unit conversions
def eV_to_wavenumber(energy_eV: float) -> float:
    """Convert energy in eV to wavenumber in cm⁻¹."""
    return energy_eV * eV_to_J / (h * c * 100)
def wavenumber_to_eV(wavenumber_cm: float) -> float:
    """Convert wavenumber in cm⁻¹ to energy in eV."""
    return wavenumber_cm * h * c * 100 / eV_to_J
def eV_to_frequency(energy_eV: float) -> float:
    """Convert energy in eV to frequency in Hz."""
    return energy_eV * eV_to_J / h
def frequency_to_eV(frequency_Hz: float) -> float:
    """Convert frequency in Hz to energy in eV."""
    return frequency_Hz * h / eV_to_J
def eV_to_wavelength(energy_eV: float) -> float:
    """Convert energy in eV to wavelength in meters."""
    return h * c / (energy_eV * eV_to_J)
def wavelength_to_eV(wavelength_m: float) -> float:
    """Convert wavelength in meters to energy in eV."""
    return h * c / (wavelength_m * eV_to_J)
# Quantum mechanical derived constants
def compton_wavelength(mass_kg: float) -> float:
    """Calculate Compton wavelength for a particle of given mass."""
    return h / (mass_kg * c)
def classical_electron_radius() -> float:
    """Calculate classical electron radius."""
    return e**2 / (4 * np.pi * epsilon0 * me * c**2)
def cyclotron_frequency(B_tesla: float, mass_kg: float = me, charge_C: float = e) -> float:
    """Calculate cyclotron frequency for charged particle in magnetic field."""
    return charge_C * B_tesla / mass_kg
def plasma_frequency(n_density: float, mass_kg: float = me, charge_C: float = e) -> float:
    """Calculate plasma frequency for given particle density."""
    return np.sqrt(n_density * charge_C**2 / (epsilon0 * mass_kg))
def debye_length(T_kelvin: float, n_density: float) -> float:
    """Calculate Debye screening length in plasma."""
    return np.sqrt(epsilon0 * kb * T_kelvin / (n_density * e**2))
def thermal_de_broglie_wavelength(T_kelvin: float, mass_kg: float = me) -> float:
    """Calculate thermal de Broglie wavelength."""
    return h / np.sqrt(2 * np.pi * mass_kg * kb * T_kelvin)
# Atomic units conversions
def convert_to_atomic_units(value: float, unit_type: str) -> float:
    """
    Convert physical quantities to atomic units.
    Parameters
    ----------
    value : float
        Value to convert
    unit_type : str
        Type of unit: 'energy', 'length', 'time', 'mass', 'charge'
    Returns
    -------
    float
        Value in atomic units
    """
    conversions = {
        'energy': lambda x: x / Eh,
        'length': lambda x: x / a0,
        'time': lambda x: x / atomic_time_to_s,
        'mass': lambda x: x / me,
        'charge': lambda x: x / e,
        'velocity': lambda x: x / (a0 / atomic_time_to_s),
        'momentum': lambda x: x / (me * a0 / atomic_time_to_s),
        'angular_momentum': lambda x: x / hbar,
        'electric_field': lambda x: x / (Eh / (e * a0)),
        'magnetic_field': lambda x: x / (hbar / (e * a0**2)),
    }
    if unit_type not in conversions:
        raise ValueError(f"Unknown unit type: {unit_type}")
    return conversions[unit_type](value)
def convert_from_atomic_units(value: float, unit_type: str) -> float:
    """
    Convert from atomic units to SI units.
    Parameters
    ----------
    value : float
        Value in atomic units
    unit_type : str
        Type of unit: 'energy', 'length', 'time', 'mass', 'charge'
    Returns
    -------
    float
        Value in SI units
    """
    conversions = {
        'energy': lambda x: x * Eh,
        'length': lambda x: x * a0,
        'time': lambda x: x * atomic_time_to_s,
        'mass': lambda x: x * me,
        'charge': lambda x: x * e,
        'velocity': lambda x: x * (a0 / atomic_time_to_s),
        'momentum': lambda x: x * (me * a0 / atomic_time_to_s),
        'angular_momentum': lambda x: x * hbar,
        'electric_field': lambda x: x * (Eh / (e * a0)),
        'magnetic_field': lambda x: x * (hbar / (e * a0**2)),
    }
    if unit_type not in conversions:
        raise ValueError(f"Unknown unit type: {unit_type}")
    return conversions[unit_type](value)