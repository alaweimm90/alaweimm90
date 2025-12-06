"""
Swarm Intelligence Voting

Collective decision-making from 100+ agents.

Usage:
    from swarm_voting import SwarmVotingProtocol

    protocol = SwarmVotingProtocol(num_agents=100)
    result = await protocol.vote(question, options)
"""

__version__ = "0.1.0"

from swarm_voting.protocol import SwarmVotingProtocol
from swarm_voting.core.models import (
    VotingResult,
    Vote,
    AgentVoter,
    ConsensusLevel,
)

__all__ = [
    "SwarmVotingProtocol",
    "VotingResult",
    "Vote",
    "AgentVoter",
    "ConsensusLevel",
]
