"""
Librex.Meta - Meta-Learning for Automatic Algorithm Selection

The "solver of solvers" - automatically selects the best Libria solver for each
optimization problem based on problem features and historical performance.

**Problem Type:** Algorithm Selection / Meta-Learning
**Use Case:** MEZAN automatically choosing optimal solver for each problem
**Target:** Minimize regret over solver choices
"""

__version__ = "1.0.0"

from .solver import Librex.MetaSolver

__all__ = ["Librex.MetaSolver"]
