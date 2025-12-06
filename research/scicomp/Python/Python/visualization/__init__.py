#!/usr/bin/env python3
"""
Scientific Visualization Module
Berkeley-themed plotting utilities for scientific computing applications.
Provides publication-quality figures with UC Berkeley's official color scheme.
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
from .berkeley_style import *
from .interactive import *
from .quantum_viz import *
from .materials_viz import *
__all__ = [
    # Berkeley styling
    'BerkeleyPlot', 'apply_berkeley_style', 'BERKELEY_COLORS',
    'publication_figure', 'save_publication_figure',
    # Interactive visualizations
    'InteractivePlot', 'create_dashboard', 'animate_wavefunction',
    'interactive_band_structure',
    # Quantum visualizations
    'plot_wavefunction', 'plot_bloch_sphere', 'plot_quantum_circuit',
    'plot_energy_levels', 'plot_probability_density',
    # Materials visualizations
    'plot_crystal_structure', 'plot_phonon_bands', 'plot_electronic_bands',
    'plot_dos', 'plot_fermi_surface'
]