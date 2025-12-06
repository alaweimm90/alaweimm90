"""
Librex: Universal Optimization Framework

A production-grade optimization library with 31+ algorithms,
GPU acceleration, and enterprise-scale performance.
"""

from __future__ import annotations

from equilibria.core.interfaces import (
    StandardizedProblem,
    StandardizedSolution,
    UniversalOptimizationInterface,
    ValidationResult,
)
from equilibria.optimize import optimize

__version__ = "1.0.0"
__author__ = "Meshal Alawein"
__email__ = "meshal@berkeley.edu"

__all__ = [
    "optimize",
    "StandardizedProblem",
    "StandardizedSolution",
    "UniversalOptimizationInterface",
    "ValidationResult",
]
