"""
Benchmarking utilities for Librex

Provides tools for measuring and comparing optimization method performance.
"""

from Librex.benchmarking.benchmark import (
    BenchmarkResult,
    benchmark_method,
    compare_methods,
    benchmark_suite,
)
from Librex.benchmarking.metrics import (
    compute_optimality_gap,
    compute_solution_quality,
    compute_convergence_speed,
)

__all__ = [
    'BenchmarkResult',
    'benchmark_method',
    'compare_methods',
    'benchmark_suite',
    'compute_optimality_gap',
    'compute_solution_quality',
    'compute_convergence_speed',
]
