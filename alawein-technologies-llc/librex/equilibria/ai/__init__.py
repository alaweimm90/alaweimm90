"""
AI-powered method selection for Librex

This module provides intelligent optimization method selection based on problem
characteristics. It uses a combination of heuristic rules and optional machine
learning models to recommend the best optimization method for a given problem.

Key components:
- features: Problem feature extraction
- method_selector: AI-powered method recommendation
- models: Machine learning models for prediction (future enhancement)

Example:
    >>> from Librex.ai import MethodSelector
    >>> from Librex.adapters.qap import QAPAdapter
    >>> import numpy as np
    >>>
    >>> problem = {
    ...     'flow_matrix': np.array([[0, 5, 3], [5, 0, 2], [3, 2, 0]]),
    ...     'distance_matrix': np.array([[0, 8, 4], [8, 0, 6], [4, 6, 0]])
    ... }
    >>> adapter = QAPAdapter()
    >>> selector = MethodSelector()
    >>> method, config, confidence = selector.recommend_method(problem, adapter)
    >>> print(f"Recommended: {method} (confidence: {confidence:.2f})")
"""

from Librex.ai.features import ProblemFeatureExtractor
from Librex.ai.method_selector import MethodSelector

__all__ = ["ProblemFeatureExtractor", "MethodSelector"]