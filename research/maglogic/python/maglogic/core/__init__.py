"""
Core functionality for MagLogic package.

This module provides fundamental constants, units, and validation functions
used throughout the MagLogic suite for computational magnetism.

Author: Meshal Alawein
Email: meshal@berkeley.edu
"""

from .constants import *
from .units import *
from .validators import *

__all__ = [
    "PHYSICAL_CONSTANTS", "MATERIAL_CONSTANTS", "SIMULATION_DEFAULTS",
    "Unit", "convert_units", "validate_input", "validate_simulation_parameters"
]