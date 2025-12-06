"""
Librex.QAP - Quadratic Assignment Problem Solver for MEZAN

This solver handles agent-task assignment problems where pairwise synergies
and conflicts between agents must be considered.

**Problem Type:** Quadratic Assignment Problem (QAP)
**Use Case:** ORCHEX agent assignment to research tasks
**Target:** 20%+ improvement over random/greedy baselines
"""

__version__ = "1.0.0"

from .solver import Librex.QAPSolver
from .algorithms import SimulatedAnnealingSolver, GeneticAlgorithmSolver
from .benchmarks import QAPLIBBenchmark

__all__ = [
    "Librex.QAPSolver",
    "SimulatedAnnealingSolver",
    "GeneticAlgorithmSolver",
    "QAPLIBBenchmark",
]
