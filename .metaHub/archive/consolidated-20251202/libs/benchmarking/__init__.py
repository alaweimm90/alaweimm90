"""
MetaHub Benchmarking Library

Universal benchmarking and performance profiling for all projects.

Consolidated from:
- organizations/AlaweinOS/Benchmarks/ (MEZAN benchmark suite)
- Future: Additional benchmarking code from other projects

Consolidation Date: 2025-01-29
Migration: See .archive/benchmarks-consolidation/MIGRATION.md

Usage:
    from metahub.libs.benchmarking import BenchmarkRunner, run_benchmark
    from metahub.libs.benchmarking.visualization import create_charts

Example:
    runner = BenchmarkRunner(output_dir="./results")
    results = runner.run_benchmarks(targets)
    runner.save_results()
"""

from .core import BenchmarkRunner, run_benchmark, format_results, load_latest_results
from .visualization import (
    create_performance_chart,
    create_summary_table,
    generate_markdown_report
)

__all__ = [
    "BenchmarkRunner",
    "run_benchmark",
    "format_results",
    "load_latest_results",
    "create_performance_chart",
    "create_summary_table",
    "generate_markdown_report",
]

__version__ = "1.0.0"
