#!/usr/bin/env python3
"""
Many-Body Quantum Physics Module
Advanced many-body quantum systems including exact diagonalization,
mean-field theory, quantum Monte Carlo methods, and entanglement analysis.
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
from .exact_diagonalization import *
from .hubbard_model import *
from .heisenberg_model import *
from .entanglement_analysis import *
from .mean_field_theory import *
__all__ = [
    # Exact Diagonalization
    'ExactDiagonalization', 'sparse_hamiltonian', 'ground_state_solver',
    'many_body_spectrum', 'correlation_functions',
    # Hubbard Model
    'HubbardModel', 'fermi_hubbard_hamiltonian', 'hubbard_physics',
    'phase_diagram', 'magnetic_properties',
    # Heisenberg Model
    'HeisenbergModel', 'spin_chain_hamiltonian', 'magnetic_excitations',
    'spin_correlations', 'quantum_magnetism',
    # Entanglement Analysis
    'EntanglementAnalyzer', 'entanglement_entropy', 'schmidt_decomposition',
    'mutual_information', 'entanglement_spectrum',
    # Mean-Field Theory
    'MeanFieldSolver', 'hartree_fock', 'bcs_theory', 'self_consistency',
    'phase_transitions'
]