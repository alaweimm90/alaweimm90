"""
Warm-Starting Infrastructure for Librex

This module provides comprehensive warm-starting capabilities for optimization,
allowing solutions from previous problems to accelerate convergence on new problems.

Key Components:
1. Solution Cache - Database for storing and retrieving past solutions
2. Solution Transformation - Methods to adapt solutions between problems
3. Warm-Start Strategies - Different approaches for using past solutions
4. Method Adapters - Integration with specific optimization methods
5. Incremental Optimization - Solving sequences of related problems

Benefits:
- Faster convergence on similar problems
- Better solution quality through transfer learning
- Efficient parameter sweeps and sensitivity analysis
- Adaptive problem solving with continuous refinement
"""

from .solution_cache import SolutionCache
from .transform import SolutionTransformer
from .strategies import WarmStartStrategy, WarmStartManager
from .method_adapters import (
    GeneticAlgorithmWarmStart,
    SimulatedAnnealingWarmStart,
    TabuSearchWarmStart,
    LocalSearchWarmStart
)
from .incremental import IncrementalOptimizer

__all__ = [
    'SolutionCache',
    'SolutionTransformer',
    'WarmStartStrategy',
    'WarmStartManager',
    'GeneticAlgorithmWarmStart',
    'SimulatedAnnealingWarmStart',
    'TabuSearchWarmStart',
    'LocalSearchWarmStart',
    'IncrementalOptimizer'
]