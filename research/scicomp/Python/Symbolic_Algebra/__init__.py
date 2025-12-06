"""
Symbolic Algebra module for SciComp.
Provides comprehensive symbolic computation tools including:
- Expression manipulation and simplification
- Symbolic differentiation and integration
- Equation solving
- Series expansions and transforms
"""
from .core.symbolic_computation import (
    SymbolicExpression,
    SymbolicMatrix,
    EquationSolver,
    SeriesExpansion,
    SymbolicIntegration,
    SymbolicTransforms
)
__all__ = [
    'SymbolicExpression',
    'SymbolicMatrix',
    'EquationSolver',
    'SeriesExpansion',
    'SymbolicIntegration',
    'SymbolicTransforms'
]
__version__ = '1.0.0'
__author__ = 'UC Berkeley SciComp Team'