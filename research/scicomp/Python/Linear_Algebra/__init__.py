"""
Linear Algebra Package for Scientific Computing
A comprehensive linear algebra library providing core operations, decompositions,
and linear system solvers with Berkeley-themed examples and visualizations.
Modules:
    core.matrix_operations: Matrix arithmetic, decompositions, and properties
    core.vector_operations: Vector operations, norms, and algorithms
    core.linear_systems: Direct and iterative linear system solvers
Examples:
    examples.beginner.basic_operations: Fundamental operations and concepts
    examples.intermediate.matrix_decompositions: Advanced decomposition techniques
    examples.advanced.iterative_methods: Large system solvers and preconditioning
"""
from .core.matrix_operations import MatrixOperations, MatrixDecompositions, SpecialMatrices
from .core.vector_operations import VectorOperations, VectorNorms
from .core.linear_systems import DirectSolvers, IterativeSolvers, LinearSystemUtils, SolverResult
__version__ = "1.0.0"
__author__ = "SciComp"
__all__ = [
    # Matrix operations
    'MatrixOperations',
    'MatrixDecompositions',
    'SpecialMatrices',
    # Vector operations
    'VectorOperations',
    'VectorNorms',
    # Linear systems
    'DirectSolvers',
    'IterativeSolvers',
    'LinearSystemUtils',
    'SolverResult'
]