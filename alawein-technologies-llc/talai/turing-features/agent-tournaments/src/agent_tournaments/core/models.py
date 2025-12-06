"""
Core data models for Agent Tournament System
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field


class TournamentFormat(str, Enum):
    """Tournament formats"""
    FREE_FOR_ALL = "free_for_all"
    ELIMINATION = "elimination"
    ROUND_ROBIN = "round_robin"
    SWISS = "swiss"
    MULTI_STAGE = "multi_stage"


class MatchResult(BaseModel):
    """Result of a single match between agents"""
    match_id: str = Field(..., description="Unique match ID")
    agent_1_id: str = Field(..., description="First agent ID")
    agent_2_id: str = Field(..., description="Second agent ID")
    agent_1_solution: Any = Field(..., description="Agent 1's solution")
    agent_2_solution: Any = Field(..., description="Agent 2's solution")
    agent_1_score: float = Field(..., description="Agent 1's score")
    agent_2_score: float = Field(..., description="Agent 2's score")
    winner_id: str = Field(..., description="Winner agent ID")
    margin: float = Field(..., description="Victory margin")
    judge_reasoning: str = Field("", description="Why this agent won")
    timestamp: datetime = Field(default_factory=datetime.now)


class ELORating(BaseModel):
    """ELO rating for an agent"""
    agent_id: str = Field(..., description="Agent ID")
    rating: float = Field(1500.0, description="Current ELO rating")
    games_played: int = Field(0, description="Total games")
    wins: int = Field(0, description="Total wins")
    losses: int = Field(0, description="Total losses")
    draws: int = Field(0, description="Total draws")
    win_rate: float = Field(0.0, description="Win percentage")

    def update_rating(self, expected_score: float, actual_score: float, k_factor: int = 32):
        """Update ELO rating after a match"""
        self.rating = self.rating + k_factor * (actual_score - expected_score)
        self.games_played += 1

        if actual_score == 1.0:
            self.wins += 1
        elif actual_score == 0.5:
            self.draws += 1
        else:
            self.losses += 1

        self.win_rate = self.wins / self.games_played if self.games_played > 0 else 0.0


class AgentRanking(BaseModel):
    """Final ranking of an agent in tournament"""
    rank: int = Field(..., description="Final rank (1 = best)")
    agent_id: str = Field(..., description="Agent ID")
    agent_name: str = Field(..., description="Agent name")
    elo_rating: float = Field(..., description="Final ELO rating")
    wins: int = Field(..., description="Total wins")
    losses: int = Field(..., description="Total losses")
    draws: int = Field(..., description="Total draws")
    total_score: float = Field(..., description="Cumulative score")
    best_solution_quality: float = Field(..., description="Best solution found")
    consistency_score: float = Field(..., description="How consistent (0-100)")


class TournamentResult(BaseModel):
    """Complete tournament result"""
    tournament_id: str = Field(..., description="Tournament ID")
    format: TournamentFormat = Field(..., description="Tournament format used")
    problem: Any = Field(..., description="Problem being solved")

    # Results
    rankings: List[AgentRanking] = Field(..., description="Final rankings")
    all_matches: List[MatchResult] = Field(..., description="All matches played")
    champion: AgentRanking = Field(..., description="Tournament winner")
    champion_solution: Any = Field(..., description="Winning solution")

    # Statistics
    total_agents: int = Field(..., description="Number of agents")
    total_matches: int = Field(..., description="Total matches played")
    total_rounds: int = Field(..., description="Total rounds")
    average_solution_quality: float = Field(..., description="Average quality")
    improvement_over_random: float = Field(..., description="% better than random")
    improvement_over_baseline: float = Field(..., description="% better than baseline")

    # Metadata
    execution_time_seconds: float = Field(..., description="Total execution time")
    models_used: List[str] = Field(default_factory=list)
    completed_at: datetime = Field(default_factory=datetime.now)

    @property
    def competitive_pressure_effect(self) -> float:
        """Measure competitive pressure improvement"""
        if self.all_matches:
            first_round_avg = sum(m.agent_1_score + m.agent_2_score
                                for m in self.all_matches[:len(self.all_matches)//4]) / (len(self.all_matches)//4) / 2
            last_round_avg = sum(m.agent_1_score + m.agent_2_score
                               for m in self.all_matches[-len(self.all_matches)//4:]) / (len(self.all_matches)//4) / 2
            return ((last_round_avg - first_round_avg) / first_round_avg * 100) if first_round_avg > 0 else 0.0
        return 0.0


class TournamentConfig(BaseModel):
    """Configuration for tournament"""
    format: TournamentFormat = Field(TournamentFormat.ELIMINATION)
    k_factor: int = Field(32, description="ELO K-factor")
    parallel_matches: bool = Field(True, description="Run matches in parallel")
    judge_model: str = Field("gpt-4", description="Model for judging")
    require_majority: bool = Field(False, description="Require majority consensus")
    time_limit_per_match: Optional[float] = Field(None, description="Time limit per match (seconds)")
    max_rounds: Optional[int] = Field(None, description="Maximum rounds")
