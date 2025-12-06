#!/usr/bin/env python3
"""
Electronic Structure Module
Advanced electronic structure calculations including density functional theory,
band structure computation, density of states, and strain engineering effects.
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
from .dft_basics import *
from .band_structure import *
from .density_of_states import *
from .strain_engineering import *
__all__ = [
    # DFT Basics
    'DFTCalculator', 'kohn_sham_solver', 'exchange_correlation_functional',
    'self_consistent_field', 'electron_density',
    # Band Structure
    'BandStructureCalculator', 'tight_binding_model', 'calculate_bands',
    'high_symmetry_path', 'plot_band_structure',
    # Density of States
    'DOSCalculator', 'calculate_dos', 'projected_dos', 'fermi_energy',
    'integration_weights',
    # Strain Engineering
    'StrainCalculator', 'apply_strain', 'deformation_potential',
    'strain_tensor', 'bandgap_vs_strain'
]