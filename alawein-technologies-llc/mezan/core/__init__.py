"""
MEZAN Core - Integration Layer

This module provides the integration layer between MEZAN V4.0.0 distributed
infrastructure and the original vision optimization solvers (Libria Suite).
"""

__version__ = "4.1.0"  # Integration layer version

from .optimizer_interface import (
    OptimizerInterface,
    OptimizationProblem,
    OptimizationResult,
    ProblemType,
    SolverStatus,
    HeuristicFallbackOptimizer,
)
from .optimizer_factory import (
    OptimizerFactory,
    FeatureFlag,
    get_optimizer_factory,
    reset_optimizer_factory,
)

__all__ = [
    "OptimizerInterface",
    "OptimizationProblem",
    "OptimizationResult",
    "ProblemType",
    "SolverStatus",
    "HeuristicFallbackOptimizer",
    "OptimizerFactory",
    "FeatureFlag",
    "get_optimizer_factory",
    "reset_optimizer_factory",
]
