"""Librex.QAP core pipeline - Orchestrates benchmark execution.

Contains the main QAPBenchmarkPipeline class that:
- Loads QAPLIB instances
- Applies novel and baseline methods
- Tracks metrics and method usage
- Returns comprehensive BenchmarkResult
"""

from .pipeline import QAPBenchmarkPipeline

__all__ = ["QAPBenchmarkPipeline"]
