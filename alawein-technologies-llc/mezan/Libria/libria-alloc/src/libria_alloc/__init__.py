"""
Librex.Alloc - Constrained Resource Allocation for MEZAN

Allocates limited resources (API credits, compute budget) across competing
agents using Thompson Sampling with budget constraints.

**Problem Type:** Resource Allocation with Constraints
**Use Case:** Allocating Claude/GPT API credits across ORCHEX agents
**Target:** Minimize regret while respecting budget
"""

__version__ = "1.0.0"

from .solver import Librex.AllocSolver

__all__ = ["Librex.AllocSolver"]
