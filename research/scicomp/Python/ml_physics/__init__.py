#!/usr/bin/env python3
"""
Physics-Informed Machine Learning Module
Cutting-edge ML techniques for scientific computing including physics-informed
neural networks (PINNs), materials property prediction, quantum machine learning,
and scientific computing acceleration.
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
from . import pinns
from . import materials_ml
from . import quantum_ml
from . import scientific_computing_ml
__all__ = [
    'pinns',
    'materials_ml',
    'quantum_ml',
    'scientific_computing_ml'
]