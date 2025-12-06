"""
Core models for Swarm Intelligence Voting
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field


class ConsensusLevel(str, Enum):
    """Consensus levels"""
    STRONG = "strong"  # >80% agreement
    MODERATE = "moderate"  # 60-80%
    WEAK = "weak"  # 40-60%
    NO_CONSENSUS = "no_consensus"  # <40%


class Vote(BaseModel):
    """A single agent's vote"""
    agent_id: str = Field(..., description="Agent ID")
    option: str = Field(..., description="Option voted for")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in vote")
    reasoning: Optional[str] = Field(None, description="Why this vote")
    timestamp: datetime = Field(default_factory=datetime.now)


class AgentVoter(BaseModel):
    """An agent that can vote"""
    agent_id: str = Field(..., description="Agent ID")
    agent_name: str = Field(..., description="Agent name")
    expertise: List[str] = Field(default_factory=list, description="Areas of expertise")
    weight: float = Field(1.0, description="Voting weight")
    past_accuracy: float = Field(0.5, description="Historical accuracy (0-1)")


class VotingResult(BaseModel):
    """Result from swarm voting"""
    question: str = Field(..., description="Question posed")
    options: List[str] = Field(..., description="Options to choose from")

    # Votes
    all_votes: List[Vote] = Field(..., description="All votes cast")
    vote_distribution: Dict[str, int] = Field(..., description="Counts per option")
    weighted_distribution: Dict[str, float] = Field(..., description="Weighted counts")

    # Results
    winning_option: str = Field(..., description="Option with most votes")
    vote_percentage: float = Field(..., description="Percentage for winner")
    consensus_level: ConsensusLevel = Field(..., description="Consensus strength")

    # Analysis
    groupthink_detected: bool = Field(False, description="Was groupthink detected?")
    diversity_score: float = Field(..., ge=0.0, le=100.0, description="Opinion diversity")
    confidence_average: float = Field(..., description="Average confidence")

    # Metadata
    total_agents: int = Field(..., description="Number of agents")
    execution_time_seconds: float = Field(..., description="Execution time")
    completed_at: datetime = Field(default_factory=datetime.now)

    @property
    def verdict(self) -> str:
        """Human-readable verdict"""
        if self.consensus_level == ConsensusLevel.STRONG:
            return f"STRONG CONSENSUS: {self.winning_option} ({self.vote_percentage:.0f}%)"
        elif self.consensus_level == ConsensusLevel.MODERATE:
            return f"MODERATE CONSENSUS: {self.winning_option} ({self.vote_percentage:.0f}%)"
        elif self.consensus_level == ConsensusLevel.WEAK:
            return f"WEAK CONSENSUS: {self.winning_option} ({self.vote_percentage:.0f}%)"
        else:
            return f"NO CLEAR CONSENSUS - Divided opinions"

    @property
    def reliability(self) -> float:
        """How reliable is this result (0-100)"""
        consensus_score = {
            ConsensusLevel.STRONG: 90,
            ConsensusLevel.MODERATE: 70,
            ConsensusLevel.WEAK: 50,
            ConsensusLevel.NO_CONSENSUS: 30
        }[self.consensus_level]

        diversity_bonus = self.diversity_score * 0.1  # Diversity is good
        groupthink_penalty = 20 if self.groupthink_detected else 0

        return min(100, max(0, consensus_score + diversity_bonus - groupthink_penalty))
