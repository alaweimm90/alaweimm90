#!/usr/bin/env python3
"""
Common Utilities Module
Provides fundamental constants, unit conversions, file I/O operations,
and parallelization helpers used throughout the SciComp package.
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
from .constants import *
from .units import *
from .file_io import *
from .parallel import *
__all__ = [
    # Physical constants
    'PHYSICAL_CONSTANTS',
    'h', 'hbar', 'c', 'e', 'me', 'mp', 'kb', 'epsilon0', 'mu0', 'NA',
    'eV_to_J', 'J_to_eV', 'Ry_to_eV', 'Ha_to_eV', 'bohr_to_m', 'm_to_bohr',
    # Unit conversions
    'UnitConverter',
    'energy_convert', 'length_convert', 'time_convert', 'mass_convert',
    # File I/O
    'save_data', 'load_data', 'save_wavefunction', 'load_wavefunction',
    'export_results', 'create_output_directory',
    # Parallelization
    'parallel_map', 'parallel_compute', 'get_optimal_workers',
    'memory_efficient_compute', 'distributed_compute',
]