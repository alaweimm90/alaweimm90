#!/usr/bin/env python3
"""
Quantum Physics Module
Comprehensive quantum mechanics implementations including time-dependent dynamics,
electronic structure theory, many-body systems, and quantum optics.
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
from . import quantum_dynamics
from . import electronic_structure
from . import many_body
from . import quantum_optics
__all__ = [
    'quantum_dynamics',
    'electronic_structure',
    'many_body',
    'quantum_optics'
]