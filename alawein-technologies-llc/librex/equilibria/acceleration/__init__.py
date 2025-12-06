"""
GPU Acceleration and Advanced Parallelization Module for Librex

This module provides high-performance computing capabilities including:
- Multi-backend GPU support (JAX, PyTorch, CuPy)
- Distributed optimization across multiple GPUs
- Memory-efficient operations for large-scale problems
- Performance profiling and optimization tools
"""

from .gpu_backend import GPUBackend, DeviceManager
from .distributed import DistributedOptimizer
from .parallel_eval import ParallelEvaluator
from .memory_efficient import MemoryEfficientOptimizer
from .profiling import PerformanceProfiler
from .amp import enable_mixed_precision, disable_mixed_precision

__all__ = [
    'GPUBackend',
    'DeviceManager',
    'DistributedOptimizer',
    'ParallelEvaluator',
    'MemoryEfficientOptimizer',
    'PerformanceProfiler',
    'enable_mixed_precision',
    'disable_mixed_precision',
]