"""
Librex.Meta - Tournament-based Meta-Learning for Multi-Agent Solver Selection

Tournament-based solver selection using Swiss-system framework with Elo ratings
and contextual bandits for optimal solver selection.

Target: AutoML Conference 2025 (Deadline: March 31, 2025)
"""

from .meta_solver import Librex.Meta
from .feature_extractor import FeatureExtractor

__version__ = "0.1.0"
__all__ = ["Librex.Meta", "FeatureExtractor"]
