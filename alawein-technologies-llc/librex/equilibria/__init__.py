"""
Librex: Universal Optimization Framework

A production-grade optimization library with 31+ algorithms,
GPU acceleration, and enterprise-scale performance.
"""

__version__ = "1.0.0"
__author__ = "Meshal Alawein"
__email__ = "meshal@berkeley.edu"

from Librex.core.interfaces import (
    StandardizedProblem,
    StandardizedSolution,
    UniversalOptimizationInterface,
    ValidationResult,
)
from Librex.optimize import optimize

__all__ = [
    "optimize",
    "StandardizedProblem",
    "StandardizedSolution",
    "UniversalOptimizationInterface",
    "ValidationResult",
]
