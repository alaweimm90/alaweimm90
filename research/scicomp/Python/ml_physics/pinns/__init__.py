#!/usr/bin/env python3
"""
Physics-Informed Neural Networks (PINNs) Module
Implementation of PINNs for solving differential equations with physics constraints.
Includes specialized architectures for quantum mechanics, heat transfer, and wave equations.
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
from .schrodinger_pinn import *
from .heat_equation_pinn import *
from .wave_equation_pinn import *
from .elasticity_pinn import *
__all__ = [
    # Schrödinger PINN
    'SchrodingerPINN', 'solve_schrodinger_pinn', 'quantum_harmonic_pinn',
    # Heat Equation PINN
    'HeatEquationPINN', 'solve_heat_equation_pinn', 'thermal_diffusion_pinn',
    # Wave Equation PINN
    'WaveEquationPINN', 'solve_wave_equation_pinn', 'acoustic_wave_pinn',
    # Elasticity PINN
    'ElasticityPINN', 'elastic_deformation_pinn', 'stress_strain_pinn'
]