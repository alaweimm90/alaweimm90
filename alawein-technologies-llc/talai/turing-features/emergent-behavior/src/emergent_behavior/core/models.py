"""
Core models for Emergent Behavior Monitoring
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field


class BehaviorType(str, Enum):
    """Types of emergent behaviors"""
    BENEFICIAL = "beneficial"
    HARMFUL = "harmful"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class EmergentPattern(BaseModel):
    """An emergent pattern detected"""
    pattern_id: str = Field(..., description="Pattern ID")
    description: str = Field(..., description="What the pattern is")
    behavior_type: BehaviorType = Field(..., description="Beneficial/harmful/neutral")
    frequency: int = Field(..., description="How often it occurs")
    agents_involved: List[str] = Field(..., description="Which agents exhibit it")
    impact_score: float = Field(..., ge=0.0, le=100.0, description="Impact (0-100)")
    first_observed: datetime = Field(..., description="When first seen")
    example_instances: List[str] = Field(default_factory=list)


class MonitoringResult(BaseModel):
    """Result from emergent behavior monitoring"""
    monitoring_duration_seconds: float = Field(..., description="How long monitored")
    total_interactions: int = Field(..., description="Total agent interactions")

    # Patterns detected
    beneficial_patterns: List[EmergentPattern] = Field(default_factory=list)
    harmful_patterns: List[EmergentPattern] = Field(default_factory=list)
    neutral_patterns: List[EmergentPattern] = Field(default_factory=list)

    # Actions taken
    amplified_patterns: List[str] = Field(default_factory=list, description="Patterns amplified")
    suppressed_patterns: List[str] = Field(default_factory=list, description="Patterns suppressed")

    # Summary
    total_patterns: int = Field(..., description="Total patterns found")
    emergence_rate: float = Field(..., description="Patterns per minute")
    beneficial_ratio: float = Field(..., description="% beneficial")

    completed_at: datetime = Field(default_factory=datetime.now)

    @property
    def verdict(self) -> str:
        """Overall verdict"""
        if len(self.harmful_patterns) > len(self.beneficial_patterns):
            return "⚠️  WARNING - More harmful than beneficial patterns"
        elif len(self.beneficial_patterns) > 0:
            return f"✅ {len(self.beneficial_patterns)} beneficial patterns detected"
        else:
            return "ℹ️  No significant emergent behaviors"

    @property
    def health_score(self) -> float:
        """System health based on emergence (0-100)"""
        if self.total_patterns == 0:
            return 50.0

        beneficial_weight = len(self.beneficial_patterns) * 10
        harmful_weight = len(self.harmful_patterns) * -15
        neutral_weight = len(self.neutral_patterns) * 2

        raw_score = 50 + beneficial_weight + harmful_weight + neutral_weight
        return max(0, min(100, raw_score))
