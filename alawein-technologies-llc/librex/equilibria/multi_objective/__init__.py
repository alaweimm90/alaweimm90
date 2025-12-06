"""
Multi-objective optimization module for Librex.

This module provides state-of-the-art multi-objective optimization algorithms
including NSGA-II, NSGA-III, and MOEA/D, along with comprehensive quality
indicators and Pareto front analysis utilities.
"""

from .core import (
    MultiObjectiveProblem,
    MultiObjectiveSolution,
    ParetoFront,
    dominates,
    fast_non_dominated_sort,
    crowding_distance,
    hypervolume,
)
from .nsga2 import NSGA2Optimizer
from .nsga3 import NSGA3Optimizer
from .moead import MOEADOptimizer
from .indicators import (
    calculate_hypervolume,
    calculate_igd,
    calculate_gd,
    calculate_spread,
    calculate_epsilon_indicator,
)

__all__ = [
    # Core
    "MultiObjectiveProblem",
    "MultiObjectiveSolution",
    "ParetoFront",
    "dominates",
    "fast_non_dominated_sort",
    "crowding_distance",
    "hypervolume",
    # Algorithms
    "NSGA2Optimizer",
    "NSGA3Optimizer",
    "MOEADOptimizer",
    # Indicators
    "calculate_hypervolume",
    "calculate_igd",
    "calculate_gd",
    "calculate_spread",
    "calculate_epsilon_indicator",
]