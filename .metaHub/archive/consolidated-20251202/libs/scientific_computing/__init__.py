"""
MetaHub Scientific Computing Library

Universal scientific computing infrastructure for all computational physics,
mathematics, and engineering modules.

Consolidated from SciComp (27 modules):
- Control, Crystallography, Elasticity, FEM, Linear_Algebra
- Machine_Learning, Monte_Carlo, Multiphysics, ODE_PDE, Optics
- Optimization, Quantum, QuantumOptics, Signal_Processing, Spintronics
- Symbolic_Algebra, Thermal_Transport, and 10 more...

Usage:
    from metahub.libs.scientific_computing.core import constants
    from metahub.libs.scientific_computing.utils import io
"""

from .core import constants

__all__ = ["constants"]
__version__ = "1.0.0"
