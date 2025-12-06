"""
Agent Tournament System

Competitive selection of best solutions through agent tournaments.

Usage:
    from agent_tournaments import TournamentProtocol, TournamentFormat

    protocol = TournamentProtocol()
    result = await protocol.run_tournament(
        agents=agents,
        problem=problem,
        format=TournamentFormat.ELIMINATION
    )
"""

__version__ = "0.1.0"

from agent_tournaments.protocol import TournamentProtocol
from agent_tournaments.core.models import (
    TournamentFormat,
    TournamentResult,
    MatchResult,
    AgentRanking,
    ELORating,
)
from agent_tournaments.core.formats import (
    FreeForAllTournament,
    EliminationTournament,
    RoundRobinTournament,
    SwissTournament,
    MultiStageTournament,
)
from agent_tournaments.strategies.elo_system import ELOSystem
from agent_tournaments.strategies.matchmaker import Matchmaker
from agent_tournaments.strategies.judge import TournamentJudge

__all__ = [
    # Main API
    "TournamentProtocol",

    # Models
    "TournamentFormat",
    "TournamentResult",
    "MatchResult",
    "AgentRanking",
    "ELORating",

    # Tournament Formats
    "FreeForAllTournament",
    "EliminationTournament",
    "RoundRobinTournament",
    "SwissTournament",
    "MultiStageTournament",

    # Strategies
    "ELOSystem",
    "Matchmaker",
    "TournamentJudge",
]
