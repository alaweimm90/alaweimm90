"""
Librex.Graph - Information-Theoretic Agent Network Topology Optimization

Optimizes communication patterns between agents in a network to minimize
information entropy and maximize coordination efficiency.

**Problem Type:** Network Topology Optimization
**Use Case:** Optimizing agent-to-agent communication in distributed ORCHEX
**Target:** Minimize communication entropy
"""

__version__ = "1.0.0"

from .solver import Librex.GraphSolver

__all__ = ["Librex.GraphSolver"]
