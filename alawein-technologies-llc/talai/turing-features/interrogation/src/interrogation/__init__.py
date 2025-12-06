"""
200-Question Interrogation Framework

Systematic hypothesis stress-testing through guided interrogation.

Usage:
    from interrogation import InterrogationProtocol
    from self_refutation import Hypothesis

    protocol = InterrogationProtocol()
    hypothesis = Hypothesis(claim="...", domain="optimization")

    result = await protocol.interrogate(hypothesis)
    print(f"Overall Score: {result.overall_score}/100")
"""

__version__ = "0.1.0"

from interrogation.protocol import InterrogationProtocol
from interrogation.question_loader import QuestionLoader
from interrogation.interrogator import Interrogator
from interrogation.scorer import InterrogationScorer
from interrogation.validator import AnswerValidator

from interrogation.core.models import (
    QuestionCategory,
    Answer,
    CategoryResult,
    InterrogationResult,
    ConsensusAnswer,
)

__all__ = [
    # Main API
    "InterrogationProtocol",

    # Components
    "QuestionLoader",
    "Interrogator",
    "InterrogationScorer",
    "AnswerValidator",

    # Models
    "QuestionCategory",
    "Answer",
    "CategoryResult",
    "InterrogationResult",
    "ConsensusAnswer",
]
