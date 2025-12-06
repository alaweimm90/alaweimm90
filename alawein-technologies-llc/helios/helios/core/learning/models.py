"""
Data models for Hall of Failures
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
import hashlib
import json


class FailureType(str, Enum):
    """Types of failures"""
    HYPOTHESIS = "hypothesis"
    EXPERIMENTAL = "experimental"
    COMPUTATIONAL = "computational"
    INTEGRATION = "integration"
    THEORETICAL = "theoretical"


class Severity(str, Enum):
    """Failure severity levels"""
    CRITICAL = "critical"  # Fundamental flaw
    MAJOR = "major"        # Significant issue
    MINOR = "minor"        # Minor problem


class Failure(BaseModel):
    """
    A recorded failure with context and lessons

    Attributes:
        id: Unique identifier
        hypothesis: The hypothesis that failed (or description)
        failure_type: Type of failure
        severity: How severe the failure is
        description: Detailed description
        context: Additional context as dict
        lessons_learned: Extracted lessons
        prevention_strategies: How to prevent this
        root_causes: Identified root causes
        similarity_hash: Hash for similarity matching
        created_at: When failure was recorded
        updated_at: Last update time
    """
    id: str = Field(default_factory=lambda: datetime.now().isoformat() + "_" + str(hash(datetime.now())))
    hypothesis: str = Field(..., description="Hypothesis or description")
    failure_type: FailureType = Field(..., description="Type of failure")
    severity: Severity = Field(Severity.MAJOR, description="Failure severity")
    description: str = Field(..., description="Detailed description")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

    # Extracted information
    lessons_learned: List[str] = Field(default_factory=list, description="Lessons extracted")
    prevention_strategies: List[str] = Field(default_factory=list, description="Prevention strategies")
    root_causes: List[str] = Field(default_factory=list, description="Root causes")

    # Metadata
    similarity_hash: str = Field(default="", description="Hash for similarity matching")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def compute_similarity_hash(self) -> str:
        """Compute hash for similarity matching"""
        # Combine key fields
        content = f"{self.failure_type}:{self.description}:{self.hypothesis}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def __post_init__(self):
        """Compute hash after initialization"""
        if not self.similarity_hash:
            self.similarity_hash = self.compute_similarity_hash()


class FailureAnalysis(BaseModel):
    """Analysis result from recording a failure"""
    failure: Failure = Field(..., description="The recorded failure")
    similar_failures: List[Failure] = Field(default_factory=list, description="Similar past failures")
    similarity_scores: List[float] = Field(default_factory=list, description="Similarity scores")
    new_insights: List[str] = Field(default_factory=list, description="New insights from this failure")
    recommended_actions: List[str] = Field(default_factory=list, description="Recommended actions")


class RiskAssessment(BaseModel):
    """Risk assessment for a hypothesis"""
    hypothesis: str = Field(..., description="Hypothesis being assessed")
    risk_level: str = Field(..., description="Overall risk level (High/Medium/Low)")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Risk score (0-1)")
    similar_failures: List[Failure] = Field(default_factory=list, description="Similar past failures")
    warnings: List[str] = Field(default_factory=list, description="Specific warnings")
    recommendations: List[str] = Field(default_factory=list, description="Risk mitigation strategies")


class FailurePattern(BaseModel):
    """Identified pattern across multiple failures"""
    pattern_name: str = Field(..., description="Name of the pattern")
    description: str = Field(..., description="Pattern description")
    failure_count: int = Field(..., description="Number of failures matching this pattern")
    failure_types: List[FailureType] = Field(..., description="Failure types involved")
    common_root_causes: List[str] = Field(..., description="Common root causes")
    prevention_strategies: List[str] = Field(..., description="How to prevent")
    example_failures: List[str] = Field(default_factory=list, description="Example failure IDs")
