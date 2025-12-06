"""
Hall of Failures - Learn from failures

Learn MORE from failures than successes.

Usage:
    from hall_of_failures import HallOfFailures, Failure

    hall = HallOfFailures()
    failure = Failure(...)

    analysis = await hall.record_failure(failure)
"""

__version__ = "0.1.0"

from hall_of_failures.protocol import HallOfFailures
from hall_of_failures.models import (
    Failure,
    FailureType,
    Severity,
    FailureAnalysis,
    RiskAssessment,
    FailurePattern,
)
from hall_of_failures.database import FailureDatabase
from hall_of_failures.classifier import FailureClassifier
from hall_of_failures.lesson_extractor import LessonExtractor
from hall_of_failures.similarity_matcher import SimilarityMatcher
from hall_of_failures.strategy_generator import StrategyGenerator

__all__ = [
    # Main API
    "HallOfFailures",

    # Models
    "Failure",
    "FailureType",
    "Severity",
    "FailureAnalysis",
    "RiskAssessment",
    "FailurePattern",

    # Components
    "FailureDatabase",
    "FailureClassifier",
    "LessonExtractor",
    "SimilarityMatcher",
    "StrategyGenerator",
]
