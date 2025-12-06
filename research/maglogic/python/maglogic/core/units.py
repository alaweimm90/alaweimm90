"""
Unit conversion utilities for micromagnetic simulations.

This module provides comprehensive unit conversion functionality for
physical quantities commonly used in computational magnetism.

Author: Meshal Alawein
Email: meshal@berkeley.edu
License: MIT
"""

import numpy as np
from typing import Union, Dict, Any, Optional, Callable
from .constants import UNIT_CONVERSIONS, PHYSICAL_CONSTANTS
import warnings

class Unit:
    """
    Represents a physical unit with conversion capabilities.
    
    This class provides a convenient interface for unit conversions
    commonly needed in micromagnetic simulations.
    
    Attributes:
        value: Numerical value
        unit: Unit string
        quantity_type: Type of physical quantity (length, time, etc.)
    
    Example:
        >>> length = Unit(100, 'nm')
        >>> print(length.to('m'))  # Convert to meters
        1e-07
        >>> field = Unit(1000, 'Oe')  
        >>> print(field.to('A/m'))  # Convert to A/m
        79577.47154594767
    """
    
    def __init__(self, value: Union[float, np.ndarray], unit: str, quantity_type: Optional[str] = None):
        """
        Initialize a Unit object.
        
        Args:
            value: Numerical value(s)
            unit: Unit string (e.g., 'nm', 'Oe', 'ps')
            quantity_type: Type of quantity (auto-detected if None)
        """
        self.value = np.asarray(value)
        self.unit = unit
        self.quantity_type = quantity_type or self._detect_quantity_type(unit)
        
        # Validate unit
        if not self._is_valid_unit(unit, self.quantity_type):
            raise ValueError(f"Invalid unit '{unit}' for quantity type '{self.quantity_type}'")
    
    def _detect_quantity_type(self, unit: str) -> str:
        """Automatically detect the quantity type from the unit string."""
        for qty_type, conversions in UNIT_CONVERSIONS.items():
            if unit in conversions:
                return qty_type
        
        # Try to infer from common unit patterns
        if unit in ['A/m', 'Oe', 'mT', 'T', 'kA/m', 'MA/m']:
            return 'magnetic_field'
        elif unit in ['emu/cm³']:
            return 'magnetization'
        elif unit.endswith('m') or unit in ['Å', 'angstrom']:
            return 'length'
        elif unit.endswith('s') or unit in ['Hz']:
            return 'time'
        elif unit in ['K', '°C', '°F']:
            return 'temperature'
        elif unit in ['J', 'eV', 'erg', 'cal', 'meV']:
            return 'energy'
        else:
            warnings.warn(f"Could not detect quantity type for unit '{unit}'. Using 'unknown'.")
            return 'unknown'
    
    def _is_valid_unit(self, unit: str, quantity_type: str) -> bool:
        """Check if a unit is valid for the given quantity type."""
        if quantity_type == 'unknown':
            return True  # Allow unknown units
        
        return quantity_type in UNIT_CONVERSIONS and unit in UNIT_CONVERSIONS[quantity_type]
    
    def to(self, target_unit: str) -> Union[float, np.ndarray]:
        """
        Convert to target unit.
        
        Args:
            target_unit: Target unit string
            
        Returns:
            Converted value(s)
            
        Raises:
            ValueError: If conversion is not possible
        """
        if self.unit == target_unit:
            return self.value
        
        # Special handling for temperature conversions
        if self.quantity_type == 'temperature':
            return self._convert_temperature(target_unit)
        
        # Get conversion factors
        if self.quantity_type not in UNIT_CONVERSIONS:
            raise ValueError(f"No conversion available for quantity type '{self.quantity_type}'")
        
        conversions = UNIT_CONVERSIONS[self.quantity_type]
        
        if self.unit not in conversions:
            raise ValueError(f"Source unit '{self.unit}' not supported for {self.quantity_type}")
        
        if target_unit not in conversions:
            raise ValueError(f"Target unit '{target_unit}' not supported for {self.quantity_type}")
        
        # Convert to base unit, then to target unit
        base_value = self.value * conversions[self.unit]
        target_value = base_value / conversions[target_unit]
        
        return target_value
    
    def _convert_temperature(self, target_unit: str) -> Union[float, np.ndarray]:
        """Handle temperature conversions with offset corrections."""
        if self.unit == target_unit:
            return self.value
        
        # Convert source to Kelvin first
        if self.unit == 'K':
            kelvin_value = self.value
        elif self.unit == '°C':
            kelvin_value = self.value + 273.15
        elif self.unit == '°F':
            kelvin_value = (self.value - 32) * 5/9 + 273.15
        else:
            raise ValueError(f"Unsupported temperature unit: {self.unit}")
        
        # Convert from Kelvin to target
        if target_unit == 'K':
            return kelvin_value
        elif target_unit == '°C':
            return kelvin_value - 273.15
        elif target_unit == '°F':
            return (kelvin_value - 273.15) * 9/5 + 32
        else:
            raise ValueError(f"Unsupported target temperature unit: {target_unit}")
    
    def to_base_unit(self) -> Union[float, np.ndarray]:
        """Convert to the base SI unit for this quantity type."""
        base_units = {
            'length': 'm',
            'time': 's', 
            'magnetic_field': 'A/m',
            'magnetization': 'A/m',
            'energy': 'J',
            'temperature': 'K'
        }
        
        if self.quantity_type in base_units:
            return self.to(base_units[self.quantity_type])
        else:
            return self.value
    
    def __str__(self) -> str:
        """String representation."""
        if np.isscalar(self.value):
            return f"{self.value} {self.unit}"
        else:
            return f"{self.value} {self.unit}"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"Unit({self.value}, '{self.unit}', quantity_type='{self.quantity_type}')"
    
    def __add__(self, other: 'Unit') -> 'Unit':
        """Add two units (must be same quantity type)."""
        if not isinstance(other, Unit):
            raise TypeError("Can only add Unit objects")
        
        if self.quantity_type != other.quantity_type:
            raise ValueError(f"Cannot add {self.quantity_type} and {other.quantity_type}")
        
        # Convert other to self's unit
        other_converted = other.to(self.unit)
        new_value = self.value + other_converted
        
        return Unit(new_value, self.unit, self.quantity_type)
    
    def __sub__(self, other: 'Unit') -> 'Unit':
        """Subtract two units."""
        if not isinstance(other, Unit):
            raise TypeError("Can only subtract Unit objects")
        
        if self.quantity_type != other.quantity_type:
            raise ValueError(f"Cannot subtract {other.quantity_type} from {self.quantity_type}")
        
        # Convert other to self's unit
        other_converted = other.to(self.unit)
        new_value = self.value - other_converted
        
        return Unit(new_value, self.unit, self.quantity_type)
    
    def __mul__(self, scalar: Union[float, int, np.ndarray]) -> 'Unit':
        """Multiply by scalar."""
        if isinstance(scalar, Unit):
            raise NotImplementedError("Unit multiplication not implemented yet")
        
        new_value = self.value * scalar
        return Unit(new_value, self.unit, self.quantity_type)
    
    def __rmul__(self, scalar: Union[float, int, np.ndarray]) -> 'Unit':
        """Right multiplication by scalar."""
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar: Union[float, int, np.ndarray]) -> 'Unit':
        """Divide by scalar."""
        if isinstance(scalar, Unit):
            raise NotImplementedError("Unit division not implemented yet")
        
        new_value = self.value / scalar
        return Unit(new_value, self.unit, self.quantity_type)

def convert_units(value: Union[float, np.ndarray], 
                 from_unit: str, 
                 to_unit: str,
                 quantity_type: Optional[str] = None) -> Union[float, np.ndarray]:
    """
    Convert between units.
    
    Args:
        value: Value(s) to convert
        from_unit: Source unit
        to_unit: Target unit
        quantity_type: Type of quantity (auto-detected if None)
        
    Returns:
        Converted value(s)
        
    Example:
        >>> convert_units(100, 'nm', 'm')
        1e-07
        >>> convert_units(1000, 'Oe', 'A/m')
        79577.47154594767
    """
    unit_obj = Unit(value, from_unit, quantity_type)
    return unit_obj.to(to_unit)

def oersted_to_am(oersted: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert Oersted to A/m."""
    return oersted * 1000.0 / (4 * np.pi)

def am_to_oersted(am: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert A/m to Oersted."""
    return am * 4 * np.pi / 1000.0

def tesla_to_am(tesla: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert Tesla to A/m."""
    mu_0 = PHYSICAL_CONSTANTS["mu_0"]
    return tesla / mu_0

def am_to_tesla(am: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert A/m to Tesla."""
    mu_0 = PHYSICAL_CONSTANTS["mu_0"]
    return am * mu_0

def emu_to_am(emu_per_cm3: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert emu/cm³ to A/m."""
    return emu_per_cm3 * 1000.0

def am_to_emu(am: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert A/m to emu/cm³."""
    return am / 1000.0

def celsius_to_kelvin(celsius: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15

def kelvin_to_celsius(kelvin: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert Kelvin to Celsius."""
    return kelvin - 273.15

def fahrenheit_to_kelvin(fahrenheit: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert Fahrenheit to Kelvin."""
    return (fahrenheit - 32) * 5/9 + 273.15

def kelvin_to_fahrenheit(kelvin: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert Kelvin to Fahrenheit."""
    return (kelvin - 273.15) * 9/5 + 32

def eV_to_joules(eV: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert electron volts to Joules."""
    return eV * PHYSICAL_CONSTANTS["e"]

def joules_to_eV(joules: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert Joules to electron volts."""
    return joules / PHYSICAL_CONSTANTS["e"]

def frequency_to_period(frequency: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert frequency (Hz) to period (s)."""
    return 1.0 / frequency

def period_to_frequency(period: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert period (s) to frequency (Hz)."""
    return 1.0 / period

def wavelength_to_frequency(wavelength: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert wavelength (m) to frequency (Hz)."""
    c = PHYSICAL_CONSTANTS["c"]
    return c / wavelength

def frequency_to_wavelength(frequency: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert frequency (Hz) to wavelength (m)."""
    c = PHYSICAL_CONSTANTS["c"]
    return c / frequency

def get_unit_info(unit: str) -> Dict[str, Any]:
    """
    Get information about a unit.
    
    Args:
        unit: Unit string
        
    Returns:
        Dictionary with unit information
    """
    info = {"unit": unit, "found": False, "quantity_types": []}
    
    # Search through all quantity types
    for qty_type, conversions in UNIT_CONVERSIONS.items():
        if unit in conversions:
            info["found"] = True
            info["quantity_types"].append(qty_type)
            info[f"{qty_type}_factor"] = conversions[unit]
    
    return info

def list_units(quantity_type: Optional[str] = None) -> Dict[str, Any]:
    """
    List available units.
    
    Args:
        quantity_type: Specific quantity type to list (all if None)
        
    Returns:
        Dictionary of available units
    """
    if quantity_type is None:
        return UNIT_CONVERSIONS.copy()
    elif quantity_type in UNIT_CONVERSIONS:
        return {quantity_type: UNIT_CONVERSIONS[quantity_type]}
    else:
        raise ValueError(f"Unknown quantity type: {quantity_type}")

def validate_unit(unit: str, quantity_type: str) -> bool:
    """
    Validate if a unit is appropriate for a quantity type.
    
    Args:
        unit: Unit string to validate
        quantity_type: Expected quantity type
        
    Returns:
        True if valid, False otherwise
    """
    if quantity_type not in UNIT_CONVERSIONS:
        return False
    
    return unit in UNIT_CONVERSIONS[quantity_type]

# Common unit conversion shortcuts
COMMON_CONVERSIONS: Dict[str, Callable] = {
    "nm_to_m": lambda x: x * 1e-9,
    "m_to_nm": lambda x: x / 1e-9,
    "ps_to_s": lambda x: x * 1e-12,
    "s_to_ps": lambda x: x / 1e-12,
    "oe_to_am": oersted_to_am,
    "am_to_oe": am_to_oersted,
    "t_to_am": tesla_to_am,
    "am_to_t": am_to_tesla,
    "c_to_k": celsius_to_kelvin,
    "k_to_c": kelvin_to_celsius,
    "ev_to_j": eV_to_joules,
    "j_to_ev": joules_to_eV,
}

def quick_convert(value: Union[float, np.ndarray], conversion: str) -> Union[float, np.ndarray]:
    """
    Quick unit conversion using predefined shortcuts.
    
    Args:
        value: Value to convert
        conversion: Conversion shortcut (e.g., 'nm_to_m', 'oe_to_am')
        
    Returns:
        Converted value
        
    Example:
        >>> quick_convert(100, 'nm_to_m')
        1e-07
    """
    if conversion not in COMMON_CONVERSIONS:
        available = list(COMMON_CONVERSIONS.keys())
        raise ValueError(f"Unknown conversion '{conversion}'. Available: {available}")
    
    return COMMON_CONVERSIONS[conversion](value)