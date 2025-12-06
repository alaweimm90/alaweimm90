"""
Librex.Flow - Confidence-Aware Workflow Routing for MEZAN

Routes research workflows through agent networks based on validation confidence
scores and quality objectives.

**Problem Type:** Workflow Routing with Confidence
**Use Case:** ORCHEX research task routing through validation agents
**Target:** Maximize end-to-end workflow confidence
"""

__version__ = "1.0.0"

from .solver import Librex.FlowSolver

__all__ = ["Librex.FlowSolver"]
