"""
Self-Refutation Protocol

Popperian Falsification for AI Hypothesis Testing

A hypothesis gains strength not by accumulating evidence FOR it,
but by surviving systematic attempts to DISPROVE it.

Usage:
    from self_refutation import SelfRefutationProtocol, Hypothesis

    protocol = SelfRefutationProtocol()
    hypothesis = Hypothesis(
        claim="Our new QAP solver achieves 10% better results",
        domain="optimization"
    )

    result = await protocol.refute(hypothesis)
    print(f"Strength: {result.strength_score}/100")
"""

__version__ = "0.1.0"

from self_refutation.protocol import SelfRefutationProtocol
from self_refutation.core.models import (
    Hypothesis,
    HypothesisDomain,
    RefutationResult,
    StrategyResult,
    RefutationStrategy,
    Confidence,
)
from self_refutation.strategies import (
    BaseRefutationStrategy,
    LogicalContradictionStrategy,
    EmpiricalCounterExampleStrategy,
    AnalogicalFalsificationStrategy,
    BoundaryViolationStrategy,
    MechanismImplausibilityStrategy,
)

__all__ = [
    # Main API
    "SelfRefutationProtocol",

    # Models
    "Hypothesis",
    "HypothesisDomain",
    "RefutationResult",
    "StrategyResult",
    "RefutationStrategy",
    "Confidence",

    # Strategies
    "BaseRefutationStrategy",
    "LogicalContradictionStrategy",
    "EmpiricalCounterExampleStrategy",
    "AnalogicalFalsificationStrategy",
    "BoundaryViolationStrategy",
    "MechanismImplausibilityStrategy",
]
