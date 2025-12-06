#!/usr/bin/env python3
"""
Quantum Computing Module
Modern quantum computing algorithms, circuits, and noise models for
variational quantum algorithms, error mitigation, and quantum simulation.
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
from . import algorithms
from . import circuits
from . import noise_models
from . import backends
__all__ = [
    'algorithms',
    'circuits',
    'noise_models',
    'backends'
]