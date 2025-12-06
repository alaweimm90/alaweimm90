#!/usr/bin/env python3
"""
Quantum Dynamics Module
Time-dependent quantum mechanics, wavepacket evolution, and tunneling phenomena.
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
from .tdse_solver import *
from .harmonic_oscillator import *
from .wavepacket_evolution import *
from .quantum_tunneling import *
__all__ = [
    # TDSE Solver
    'TDSESolver', 'CrankNicolsonSolver', 'SplitOperatorSolver',
    'evolve_wavefunction', 'schrodinger_evolution',
    # Harmonic Oscillator
    'QuantumHarmonic', 'harmonic_eigenstate', 'coherent_state',
    'harmonic_time_evolution', 'wigner_function',
    # Wavepacket Evolution
    'GaussianWavepacket', 'evolve_wavepacket', 'calculate_dispersion',
    'group_velocity', 'phase_velocity',
    # Quantum Tunneling
    'TunnelingBarrier', 'transmission_coefficient', 'reflection_coefficient',
    'tunneling_time', 'wkb_approximation'
]