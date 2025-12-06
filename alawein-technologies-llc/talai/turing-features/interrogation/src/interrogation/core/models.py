"""
Core data models for Interrogation Framework
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field


class QuestionCategory(BaseModel):
    """A category of questions"""
    id: int = Field(..., description="Category ID")
    name: str = Field(..., description="Category name")
    weight: float = Field(..., description="Category weight for scoring")
    description: str = Field(..., description="Category description")
    question_count: int = Field(..., description="Number of questions")
    questions: List[str] = Field(..., description="List of questions")


class Answer(BaseModel):
    """Answer to a single question"""
    question: str = Field(..., description="The question")
    answer: str = Field(..., description="The answer text")
    model: str = Field(..., description="Model that generated answer")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Answer confidence (0-1)")
    tokens_used: int = Field(0, description="Tokens used for this answer")
    timestamp: datetime = Field(default_factory=datetime.now)


class CategoryResult(BaseModel):
    """Result for a single category"""
    category_name: str = Field(..., description="Category name")
    category_weight: float = Field(..., description="Category weight")
    raw_score: float = Field(..., ge=0.0, le=100.0, description="Unweighted score")
    weighted_score: float = Field(..., ge=0.0, le=100.0, description="Weighted score")
    questions_asked: int = Field(..., description="Number of questions asked")
    answers: List[Answer] = Field(default_factory=list, description="All answers")
    strengths: List[str] = Field(default_factory=list, description="Identified strengths")
    weaknesses: List[str] = Field(default_factory=list, description="Identified weaknesses")
    key_issues: List[str] = Field(default_factory=list, description="Critical issues")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class InterrogationResult(BaseModel):
    """Complete interrogation result"""
    hypothesis: Any = Field(..., description="The hypothesis that was interrogated")
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Overall weighted score")
    category_results: List[CategoryResult] = Field(..., description="Results per category")

    # Categorized results
    strong_categories: List[str] = Field(default_factory=list, description="Categories scoring ≥80")
    adequate_categories: List[str] = Field(default_factory=list, description="Categories scoring 60-79")
    weak_categories: List[str] = Field(default_factory=list, description="Categories scoring 40-59")
    critical_categories: List[str] = Field(default_factory=list, description="Categories scoring <40")

    # Analysis
    failure_points: List[str] = Field(default_factory=list, description="Critical weaknesses")
    recommendations: List[str] = Field(default_factory=list, description="Actionable recommendations")

    # Metadata
    total_questions: int = Field(..., description="Total questions asked")
    total_tokens: int = Field(0, description="Total tokens used")
    execution_time_seconds: float = Field(0.0, description="Execution time")
    models_used: List[str] = Field(default_factory=list, description="Models used")
    consensus_used: bool = Field(False, description="Whether consensus scoring was used")

    completed_at: datetime = Field(default_factory=datetime.now)

    @property
    def interpretation(self) -> str:
        """Human-readable interpretation"""
        if self.overall_score >= 85:
            return "Excellent - Ready for publication"
        elif self.overall_score >= 70:
            return "Good - Minor improvements needed"
        elif self.overall_score >= 55:
            return "Adequate - Moderate concerns"
        elif self.overall_score >= 40:
            return "Weak - Major issues"
        else:
            return "Poor - Critical flaws"

    @property
    def status_emoji(self) -> str:
        """Emoji representing status"""
        if self.overall_score >= 85:
            return "🟢"
        elif self.overall_score >= 70:
            return "🟡"
        elif self.overall_score >= 55:
            return "🟠"
        else:
            return "🔴"


class ConsensusAnswer(BaseModel):
    """Consensus answer from multiple models"""
    question: str = Field(..., description="The question")
    answers: List[Answer] = Field(..., description="Answers from different models")
    consensus_answer: str = Field(..., description="Synthesized consensus answer")
    agreement_score: float = Field(..., ge=0.0, le=1.0, description="How much models agree")
    final_confidence: float = Field(..., ge=0.0, le=1.0, description="Final confidence")

    def get_best_answer(self) -> Answer:
        """Get the highest confidence answer"""
        return max(self.answers, key=lambda a: a.confidence)
