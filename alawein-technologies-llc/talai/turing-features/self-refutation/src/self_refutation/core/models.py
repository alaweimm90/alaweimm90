"""
Core data models for Self-Refutation Protocol
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field


class HypothesisDomain(str, Enum):
    """Domain of the hypothesis"""
    OPTIMIZATION = "optimization"
    MACHINE_LEARNING = "machine_learning"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    MATHEMATICS = "mathematics"
    COMPUTER_SCIENCE = "computer_science"
    GENERAL = "general"


class Confidence(str, Enum):
    """Confidence level in refutation result"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class Hypothesis(BaseModel):
    """
    A scientific hypothesis to be tested

    Attributes:
        claim: The main claim being made
        domain: Scientific domain
        context: Additional context about the hypothesis
        evidence: Supporting evidence (if any)
        assumptions: Underlying assumptions
        predictions: Testable predictions
        metadata: Additional metadata
    """
    claim: str = Field(..., description="The hypothesis claim", min_length=10)
    domain: HypothesisDomain = Field(HypothesisDomain.GENERAL, description="Scientific domain")
    context: Optional[str] = Field(None, description="Additional context")
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence")
    assumptions: List[str] = Field(default_factory=list, description="Underlying assumptions")
    predictions: List[str] = Field(default_factory=list, description="Testable predictions")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    # Internal
    hypothesis_id: Optional[str] = Field(None, description="Unique identifier")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")


class RefutationStrategy(str, Enum):
    """Types of refutation strategies"""
    LOGICAL_CONTRADICTION = "logical_contradiction"
    EMPIRICAL_COUNTER_EXAMPLE = "empirical_counter_example"
    ANALOGICAL_FALSIFICATION = "analogical_falsification"
    BOUNDARY_VIOLATION = "boundary_violation"
    MECHANISM_IMPLAUSIBILITY = "mechanism_implausibility"


class StrategyResult(BaseModel):
    """
    Result from a single refutation strategy

    Attributes:
        strategy: Which strategy was used
        passed: Whether hypothesis survived this strategy
        confidence: Confidence in the result
        reasoning: Explanation of the result
        evidence: Evidence found (if any)
        severity: How severe the refutation is (0-1)
        metadata: Additional data
    """
    strategy: RefutationStrategy = Field(..., description="Strategy used")
    passed: bool = Field(..., description="Whether hypothesis passed")
    confidence: Confidence = Field(..., description="Confidence level")
    reasoning: str = Field(..., description="Explanation")
    evidence: List[str] = Field(default_factory=list, description="Evidence found")
    severity: float = Field(0.0, ge=0.0, le=1.0, description="Refutation severity (0=none, 1=critical)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional data")


class RefutationResult(BaseModel):
    """
    Complete result of refutation protocol

    Attributes:
        hypothesis: The hypothesis that was tested
        strength_score: Overall strength score (0-100)
        strategies_passed: Number of strategies passed
        total_strategies: Total number of strategies run
        refuted: Whether hypothesis was refuted
        refutation_reason: Reason for refutation (if refuted)
        confidence: Overall confidence in result
        strategy_results: Individual strategy results
        recommendations: Recommendations for next steps
        metadata: Additional result data
    """
    hypothesis: Hypothesis = Field(..., description="Tested hypothesis")
    strength_score: float = Field(..., ge=0.0, le=100.0, description="Strength score (0-100)")
    strategies_passed: int = Field(..., ge=0, description="Strategies passed")
    total_strategies: int = Field(..., ge=1, description="Total strategies run")
    refuted: bool = Field(..., description="Whether hypothesis was refuted")
    refutation_reason: Optional[str] = Field(None, description="Reason for refutation")
    confidence: Confidence = Field(..., description="Overall confidence")
    strategy_results: List[StrategyResult] = Field(..., description="Individual strategy results")
    recommendations: List[str] = Field(default_factory=list, description="Next steps")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional data")
    completed_at: datetime = Field(default_factory=datetime.now, description="Completion timestamp")

    @property
    def pass_rate(self) -> float:
        """Percentage of strategies passed"""
        if self.total_strategies == 0:
            return 0.0
        return (self.strategies_passed / self.total_strategies) * 100

    @property
    def interpretation(self) -> str:
        """Human-readable interpretation of strength score"""
        if self.strength_score >= 81:
            return "Strong hypothesis - Proceed to experiment"
        elif self.strength_score >= 61:
            return "Minor concerns - Proceed with caution"
        elif self.strength_score >= 41:
            return "Moderate concerns - Revision recommended"
        elif self.strength_score >= 21:
            return "Major issues - Major revision needed"
        else:
            return "Critically flawed - Reject immediately"
