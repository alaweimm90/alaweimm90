"""
TalAI Reproducibility Verification System

Comprehensive system for ensuring computational reproducibility of TalAI results,
including environment capture, dependency tracking, and result verification.
"""

from .reproducibility_engine import ReproducibilityEngine
from .environment_capture import EnvironmentCapture
from .dependency_tracker import DependencyTracker
from .result_verifier import ResultVerifier
from .reproducibility_badges import ReproducibilityBadges

__version__ = "1.0.0"

__all__ = [
    "ReproducibilityEngine",
    "EnvironmentCapture",
    "DependencyTracker",
    "ResultVerifier",
    "ReproducibilityBadges",
]