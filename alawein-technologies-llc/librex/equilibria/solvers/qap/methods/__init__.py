"""Librex.QAP optimization methods - Novel and Baseline implementations.

This module provides:
- novel: 11 novel optimization methods (FFT-Laplace, Reverse-Time, etc.)
- baselines: Standard baseline methods (Sinkhorn, Hungarian, 2-Opt)

Each method follows the CODEX pattern:
    result, log = method(...)  # Returns (result, MethodLog) tuple
"""

from . import baselines, novel

__all__ = ["novel", "baselines"]
