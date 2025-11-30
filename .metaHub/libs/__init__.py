"""
MetaHub Shared Libraries

Central hub for all shared functionality across organizations.

This is the control center - projects consume from here.

Modules:
    benchmarking - Performance benchmarking and profiling
    optimization - Optimization algorithms and solvers
    common - Common utilities and helpers

Usage:
    from metahub.libs.benchmarking import BenchmarkRunner
    from metahub.libs.common import logger

Architecture:
    Hub-Spoke Pattern
    - Hub (.metaHub/): Core functionality (this)
    - Spokes (organizations/*/): Thin wrappers consuming hub
    - Archive (.archive/): Preserved development history
"""

__version__ = "1.0.0"
__all__ = ["benchmarking", "common"]
