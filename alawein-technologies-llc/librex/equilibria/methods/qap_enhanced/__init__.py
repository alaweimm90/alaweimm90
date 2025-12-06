"""
FFT-Laplace Preconditioning for Quadratic Assignment Problem

This module provides advanced preconditioning techniques for QAP optimization
using FFT-based Laplace matrix decomposition and spectral ordering methods.

Mathematical Foundation:
The QAP objective is: min trace(F @ P @ D @ P^T)

When F or D are graph Laplacians, we can exploit spectral properties:
- Laplacian matrices have eigendecomposition via FFT for grid graphs
- Spectral ordering provides high-quality initial solutions
- Fast matrix-vector products reduce computational complexity

Components:
- laplace_detector: Detect and characterize Laplacian structure
- spectral_ordering: Generate initial permutations from spectral properties
- fft_ops: FFT-accelerated matrix operations
- graph_utils: Graph structure detection and exploitation
- qap_preconditioned: Main preconditioned solver integration

Theory Reference:
The 2D grid Laplacian can be written as:
L = I_n ⊗ T_n + T_n ⊗ I_n

where T_n is the 1D tridiagonal matrix with eigenvalues:
λ_k = 2(1 - cos(πk/n))

This structure enables O(n² log n) operations instead of O(n⁴).
"""

from .laplace_detector import LaplacianDetector
from .spectral_ordering import SpectralOrdering
from .fft_ops import FFTOperations
from .graph_utils import GraphUtilities
from .qap_preconditioned import PreconditionedQAPSolver

__all__ = [
    'LaplacianDetector',
    'SpectralOrdering',
    'FFTOperations',
    'GraphUtilities',
    'PreconditionedQAPSolver'
]