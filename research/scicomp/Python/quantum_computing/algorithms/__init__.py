#!/usr/bin/env python3
"""
Quantum Algorithms Module
Implementation of key quantum algorithms including VQE, QAOA, Grover's search,
and Shor's factoring algorithm with optimization and error mitigation.
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
from .vqe import *
from .qaoa import *
from .grovers import *
from .shors import *
__all__ = [
    # VQE
    'VQE', 'variational_quantum_eigensolver', 'ansatz_circuit',
    'hardware_efficient_ansatz', 'uccsd_ansatz',
    # QAOA
    'QAOA', 'quantum_approximate_optimization', 'qaoa_circuit',
    'mixer_hamiltonian', 'cost_hamiltonian',
    # Grover's Algorithm
    'GroverSearch', 'grovers_algorithm', 'oracle_circuit',
    'diffusion_operator', 'amplitude_amplification',
    # Shor's Algorithm
    'ShorFactoring', 'shors_algorithm', 'quantum_fourier_transform',
    'modular_exponentiation', 'period_finding'
]