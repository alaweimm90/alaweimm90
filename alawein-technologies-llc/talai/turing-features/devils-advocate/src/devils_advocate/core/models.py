"""
Core models for Devil's Advocate
"""

from enum import Enum
from typing import List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field


class AttackStrategy(str, Enum):
    """Types of attack strategies"""
    EDGE_CASE = "edge_case"
    ASSUMPTION = "assumption"
    SCALING = "scaling"
    COMPOSITION = "composition"
    TEMPORAL = "temporal"


class Flaw(BaseModel):
    """A flaw found in the hypothesis"""
    flaw_id: str = Field(..., description="Unique flaw ID")
    description: str = Field(..., description="What the flaw is")
    severity: str = Field(..., description="critical|high|medium|low")
    attack_strategy: AttackStrategy = Field(..., description="How it was found")
    example: Optional[str] = Field(None, description="Example demonstrating the flaw")
    mitigation: Optional[str] = Field(None, description="How to fix it")
    timestamp: datetime = Field(default_factory=datetime.now)


class EdgeCase(BaseModel):
    """An edge case to consider"""
    case_id: str = Field(..., description="Unique case ID")
    description: str = Field(..., description="What the edge case is")
    input_conditions: str = Field(..., description="Conditions that trigger it")
    expected_behavior: str = Field(..., description="What should happen")
    actual_behavior: Optional[str] = Field(None, description="What actually happens")
    probability: float = Field(..., ge=0.0, le=1.0, description="Likelihood of occurrence")


class AttackResult(BaseModel):
    """Result from Devil's Advocate attack"""
    hypothesis: Any = Field(..., description="Hypothesis that was attacked")

    # Flaws by severity
    total_flaws_found: int = Field(..., description="Total flaws")
    critical_flaws: List[Flaw] = Field(default_factory=list)
    high_flaws: List[Flaw] = Field(default_factory=list)
    medium_flaws: List[Flaw] = Field(default_factory=list)
    low_flaws: List[Flaw] = Field(default_factory=list)

    # Edge cases
    edge_cases: List[EdgeCase] = Field(default_factory=list)

    # Metadata
    attack_strategies_used: List[AttackStrategy] = Field(default_factory=list)
    iterations: int = Field(..., description="Number of attack iterations")
    execution_time_seconds: float = Field(..., description="Execution time")
    summary: str = Field("", description="Summary of findings")

    completed_at: datetime = Field(default_factory=datetime.now)

    @property
    def verdict(self) -> str:
        """Overall verdict"""
        if len(self.critical_flaws) > 0:
            return "REJECT - Critical flaws found"
        elif len(self.high_flaws) >= 3:
            return "MAJOR REVISION - Multiple high-severity issues"
        elif len(self.high_flaws) > 0:
            return "REVISION - Some issues to address"
        elif len(self.medium_flaws) > 0:
            return "MINOR REVISION - Minor issues"
        else:
            return "ACCEPT - No significant flaws"

    @property
    def robustness_score(self) -> float:
        """Score 0-100 for hypothesis robustness"""
        total_issues = len(self.critical_flaws) * 10 + len(self.high_flaws) * 5 + len(self.medium_flaws) * 2 + len(self.low_flaws)
        max_score = 100
        penalty = min(total_issues * 5, max_score)
        return max(0, max_score - penalty)
