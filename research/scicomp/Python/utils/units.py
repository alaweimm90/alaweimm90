#!/usr/bin/env python3
"""
Unit Conversion Module
Provides comprehensive unit conversion capabilities for scientific computing,
including energy, length, time, mass, and other physical quantities.
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
import numpy as np
from typing import Dict, Union, Optional
from .constants import *
class UnitConverter:
    """
    Comprehensive unit conversion utility for scientific computing.
    Supports conversion between various units commonly used in physics,
    chemistry, and materials science.
    """
    def __init__(self):
        """Initialize the unit converter with predefined conversion factors."""
        self._energy_conversions = self._init_energy_conversions()
        self._length_conversions = self._init_length_conversions()
        self._time_conversions = self._init_time_conversions()
        self._mass_conversions = self._init_mass_conversions()
        self._temperature_conversions = self._init_temperature_conversions()
        self._pressure_conversions = self._init_pressure_conversions()
        self._magnetic_field_conversions = self._init_magnetic_field_conversions()
    def _init_energy_conversions(self) -> Dict[str, float]:
        """Initialize energy conversion factors to Joules."""
        return {
            'J': 1.0,
            'eV': eV_to_J,
            'meV': eV_to_J * 1e-3,
            'keV': eV_to_J * 1e3,
            'MeV': eV_to_J * 1e6,
            'GeV': eV_to_J * 1e9,
            'Ry': Ry_to_eV * eV_to_J,
            'Ha': Ha_to_eV * eV_to_J,
            'Hartree': Ha_to_eV * eV_to_J,
            'erg': 1e-7,
            'cal': 4.184,
            'kcal': 4184.0,
            'kJ': 1000.0,
            'cm-1': h * c * 100,  # wavenumber
            'Hz': h,
            'THz': h * 1e12,
            'K': kb,  # thermal energy
            'mK': kb * 1e-3,
            'kBT_300K': kb * 300,
        }
    def _init_length_conversions(self) -> Dict[str, float]:
        """Initialize length conversion factors to meters."""
        return {
            'm': 1.0,
            'cm': 1e-2,
            'mm': 1e-3,
            'μm': 1e-6,
            'micron': 1e-6,
            'nm': 1e-9,
            'pm': 1e-12,
            'fm': 1e-15,
            'Å': 1e-10,
            'angstrom': 1e-10,
            'bohr': a0,
            'a0': a0,
            'km': 1e3,
            'ft': 0.3048,
            'in': 0.0254,
            'mil': 0.0254e-3,
        }
    def _init_time_conversions(self) -> Dict[str, float]:
        """Initialize time conversion factors to seconds."""
        return {
            's': 1.0,
            'ms': 1e-3,
            'μs': 1e-6,
            'ns': 1e-9,
            'ps': 1e-12,
            'fs': 1e-15,
            'as': 1e-18,
            'min': 60.0,
            'hr': 3600.0,
            'day': 86400.0,
            'year': 365.25 * 86400.0,
            'au_time': atomic_time_to_s,
        }
    def _init_mass_conversions(self) -> Dict[str, float]:
        """Initialize mass conversion factors to kilograms."""
        return {
            'kg': 1.0,
            'g': 1e-3,
            'mg': 1e-6,
            'μg': 1e-9,
            'ng': 1e-12,
            'u': PHYSICAL_CONSTANTS['atomic_mass_unit']['value'],
            'amu': PHYSICAL_CONSTANTS['atomic_mass_unit']['value'],
            'Da': PHYSICAL_CONSTANTS['atomic_mass_unit']['value'],
            'me': me,
            'mp': mp,
            'mn': PHYSICAL_CONSTANTS['neutron_mass']['value'],
            'lb': 0.45359237,
            'oz': 0.0283495231,
        }
    def _init_temperature_conversions(self) -> Dict[str, Dict]:
        """Initialize temperature conversion functions."""
        return {
            'K': {'offset': 0.0, 'scale': 1.0},
            'C': {'offset': 273.15, 'scale': 1.0},
            'F': {'offset': 459.67, 'scale': 5.0/9.0},
            'R': {'offset': 0.0, 'scale': 5.0/9.0},  # Rankine
        }
    def _init_pressure_conversions(self) -> Dict[str, float]:
        """Initialize pressure conversion factors to Pascals."""
        return {
            'Pa': 1.0,
            'kPa': 1e3,
            'MPa': 1e6,
            'GPa': 1e9,
            'bar': 1e5,
            'mbar': 1e2,
            'atm': 101325.0,
            'Torr': 133.322,
            'mmHg': 133.322,
            'psi': 6894.76,
            'ksi': 6.89476e6,
        }
    def _init_magnetic_field_conversions(self) -> Dict[str, float]:
        """Initialize magnetic field conversion factors to Tesla."""
        return {
            'T': 1.0,
            'mT': 1e-3,
            'μT': 1e-6,
            'nT': 1e-9,
            'G': 1e-4,  # Gauss
            'mG': 1e-7,
            'kG': 0.1,
            'Oe': 1e-4 / (4*np.pi*1e-7) * 4*np.pi*1e-7,  # Oersted (approximately)
        }
    def convert(self, value: Union[float, np.ndarray],
                from_unit: str, to_unit: str,
                quantity_type: str) -> Union[float, np.ndarray]:
        """
        Convert between units of the same physical quantity.
        Parameters
        ----------
        value : float or array
            Value(s) to convert
        from_unit : str
            Source unit
        to_unit : str
            Target unit
        quantity_type : str
            Type of physical quantity ('energy', 'length', 'time', 'mass',
            'temperature', 'pressure', 'magnetic_field')
        Returns
        -------
        float or array
            Converted value(s)
        """
        if quantity_type == 'energy':
            return self._convert_energy(value, from_unit, to_unit)
        elif quantity_type == 'length':
            return self._convert_length(value, from_unit, to_unit)
        elif quantity_type == 'time':
            return self._convert_time(value, from_unit, to_unit)
        elif quantity_type == 'mass':
            return self._convert_mass(value, from_unit, to_unit)
        elif quantity_type == 'temperature':
            return self._convert_temperature(value, from_unit, to_unit)
        elif quantity_type == 'pressure':
            return self._convert_pressure(value, from_unit, to_unit)
        elif quantity_type == 'magnetic_field':
            return self._convert_magnetic_field(value, from_unit, to_unit)
        else:
            raise ValueError(f"Unknown quantity type: {quantity_type}")
    def _convert_energy(self, value: Union[float, np.ndarray],
                       from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
        """Convert energy units."""
        if from_unit not in self._energy_conversions:
            raise ValueError(f"Unknown energy unit: {from_unit}")
        if to_unit not in self._energy_conversions:
            raise ValueError(f"Unknown energy unit: {to_unit}")
        # Convert to Joules first, then to target unit
        joules = value * self._energy_conversions[from_unit]
        return joules / self._energy_conversions[to_unit]
    def _convert_length(self, value: Union[float, np.ndarray],
                       from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
        """Convert length units."""
        if from_unit not in self._length_conversions:
            raise ValueError(f"Unknown length unit: {from_unit}")
        if to_unit not in self._length_conversions:
            raise ValueError(f"Unknown length unit: {to_unit}")
        # Convert to meters first, then to target unit
        meters = value * self._length_conversions[from_unit]
        return meters / self._length_conversions[to_unit]
    def _convert_time(self, value: Union[float, np.ndarray],
                     from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
        """Convert time units."""
        if from_unit not in self._time_conversions:
            raise ValueError(f"Unknown time unit: {from_unit}")
        if to_unit not in self._time_conversions:
            raise ValueError(f"Unknown time unit: {to_unit}")
        # Convert to seconds first, then to target unit
        seconds = value * self._time_conversions[from_unit]
        return seconds / self._time_conversions[to_unit]
    def _convert_mass(self, value: Union[float, np.ndarray],
                     from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
        """Convert mass units."""
        if from_unit not in self._mass_conversions:
            raise ValueError(f"Unknown mass unit: {from_unit}")
        if to_unit not in self._mass_conversions:
            raise ValueError(f"Unknown mass unit: {to_unit}")
        # Convert to kilograms first, then to target unit
        kilograms = value * self._mass_conversions[from_unit]
        return kilograms / self._mass_conversions[to_unit]
    def _convert_temperature(self, value: Union[float, np.ndarray],
                           from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
        """Convert temperature units."""
        if from_unit not in self._temperature_conversions:
            raise ValueError(f"Unknown temperature unit: {from_unit}")
        if to_unit not in self._temperature_conversions:
            raise ValueError(f"Unknown temperature unit: {to_unit}")
        # Convert to Kelvin first
        from_conv = self._temperature_conversions[from_unit]
        kelvin = (value + from_conv['offset']) * from_conv['scale']
        # Convert from Kelvin to target unit
        to_conv = self._temperature_conversions[to_unit]
        return kelvin / to_conv['scale'] - to_conv['offset']
    def _convert_pressure(self, value: Union[float, np.ndarray],
                         from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
        """Convert pressure units."""
        if from_unit not in self._pressure_conversions:
            raise ValueError(f"Unknown pressure unit: {from_unit}")
        if to_unit not in self._pressure_conversions:
            raise ValueError(f"Unknown pressure unit: {to_unit}")
        # Convert to Pascals first, then to target unit
        pascals = value * self._pressure_conversions[from_unit]
        return pascals / self._pressure_conversions[to_unit]
    def _convert_magnetic_field(self, value: Union[float, np.ndarray],
                               from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
        """Convert magnetic field units."""
        if from_unit not in self._magnetic_field_conversions:
            raise ValueError(f"Unknown magnetic field unit: {from_unit}")
        if to_unit not in self._magnetic_field_conversions:
            raise ValueError(f"Unknown magnetic field unit: {to_unit}")
        # Convert to Tesla first, then to target unit
        tesla = value * self._magnetic_field_conversions[from_unit]
        return tesla / self._magnetic_field_conversions[to_unit]
# Convenience functions for common conversions
def energy_convert(value: Union[float, np.ndarray],
                  from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
    """Convert energy units."""
    converter = UnitConverter()
    return converter.convert(value, from_unit, to_unit, 'energy')
def length_convert(value: Union[float, np.ndarray],
                  from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
    """Convert length units."""
    converter = UnitConverter()
    return converter.convert(value, from_unit, to_unit, 'length')
def time_convert(value: Union[float, np.ndarray],
                from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
    """Convert time units."""
    converter = UnitConverter()
    return converter.convert(value, from_unit, to_unit, 'time')
def mass_convert(value: Union[float, np.ndarray],
                from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
    """Convert mass units."""
    converter = UnitConverter()
    return converter.convert(value, from_unit, to_unit, 'mass')
def temperature_convert(value: Union[float, np.ndarray],
                       from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
    """Convert temperature units."""
    converter = UnitConverter()
    return converter.convert(value, from_unit, to_unit, 'temperature')
def pressure_convert(value: Union[float, np.ndarray],
                    from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
    """Convert pressure units."""
    converter = UnitConverter()
    return converter.convert(value, from_unit, to_unit, 'pressure')
def magnetic_field_convert(value: Union[float, np.ndarray],
                          from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
    """Convert magnetic field units."""
    converter = UnitConverter()
    return converter.convert(value, from_unit, to_unit, 'magnetic_field')