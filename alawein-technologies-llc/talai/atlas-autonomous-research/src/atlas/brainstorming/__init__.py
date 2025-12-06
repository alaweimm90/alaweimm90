"""
ORCHEX Brainstorming Module

Advanced idea generation using multiple creativity techniques.

Components:
- BrainstormEngine: Multi-strategy brainstorming
- BrainstormStrategy: Different creativity techniques
- BrainstormIdea: Generated ideas with scores
- BrainstormSession: Results from brainstorming

Cycle 23-24: Brainstorming Module
"""

__version__ = "0.1.0"

from ORCHEX.brainstorming.brainstorm_engine import (
    BrainstormEngine,
    BrainstormStrategy,
    BrainstormIdea,
    BrainstormSession
)

__all__ = [
    "BrainstormEngine",
    "BrainstormStrategy",
    "BrainstormIdea",
    "BrainstormSession",
]
