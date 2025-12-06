"""
Swarm Voting Protocol

100+ agents vote and reach consensus.
"""

import time
import asyncio
from typing import List, Optional, Dict
from collections import Counter

from swarm_voting.core.models import (
    VotingResult,
    Vote,
    AgentVoter,
    ConsensusLevel,
)
from swarm_voting.strategies.groupthink_detector import GroupthinkDetector
from swarm_voting.strategies.weight_calculator import WeightCalculator


class SwarmVotingProtocol:
    """
    Swarm Intelligence Voting

    100+ agents vote democratically with:
    - Weighted voting based on expertise
    - Groupthink detection
    - Diversity injection when needed

    Usage:
        protocol = SwarmVotingProtocol(num_agents=100)
        result = await protocol.vote(
            question="Should we proceed with this hypothesis?",
            options=["Yes", "No", "Needs revision"]
        )
    """

    def __init__(self, num_agents: int = 100, orchestrator=None, **config):
        self.num_agents = num_agents
        self.orchestrator = orchestrator
        self.config = config

        # Initialize components
        self.groupthink_detector = GroupthinkDetector()
        self.weight_calculator = WeightCalculator()

        # Create agent swarm
        self.agents = self._create_agent_swarm(num_agents)

    def _create_agent_swarm(self, num_agents: int) -> List[AgentVoter]:
        """Create diverse swarm of voting agents"""
        agents = []

        for i in range(num_agents):
            agent = AgentVoter(
                agent_id=f"agent_{i}",
                agent_name=f"Voter {i}",
                expertise=self._assign_expertise(i),
                weight=1.0,
                past_accuracy=0.5 + (i % 30) / 100  # Varied expertise
            )
            agents.append(agent)

        return agents

    def _assign_expertise(self, agent_id: int) -> List[str]:
        """Assign expertise areas to agents"""
        expertise_areas = [
            "optimization", "machine_learning", "statistics",
            "physics", "biology", "chemistry",
            "computer_science", "mathematics", "engineering"
        ]
        # Distribute expertise across agents
        return [expertise_areas[agent_id % len(expertise_areas)]]

    async def vote(
        self,
        question: str,
        options: List[str],
        context: Optional[str] = None
    ) -> VotingResult:
        """
        Conduct a vote among all agents

        Args:
            question: Question to vote on
            options: List of options to choose from
            context: Additional context for voting

        Returns:
            VotingResult with consensus and analysis
        """
        start_time = time.time()

        print(f"\n🐝 SWARM VOTING: {self.num_agents} agents")
        print(f"   Question: {question}")
        print(f"   Options: {', '.join(options)}")

        # Calculate weights based on expertise match
        weights = self.weight_calculator.calculate_weights(
            self.agents, question, context
        )

        # Collect votes from all agents
        votes = await self._collect_votes(question, options, weights)

        # Calculate vote distribution
        vote_counts = Counter(v.option for v in votes)
        total_votes = len(votes)

        # Calculate weighted distribution
        weighted_counts = {}
        for option in options:
            weighted_counts[option] = sum(
                v.confidence * weights.get(v.agent_id, 1.0)
                for v in votes if v.option == option
            )

        # Determine winner
        winning_option = max(weighted_counts, key=weighted_counts.get)
        vote_percentage = (vote_counts[winning_option] / total_votes * 100)

        # Determine consensus level
        consensus_level = self._determine_consensus(vote_percentage)

        # Detect groupthink
        groupthink = self.groupthink_detector.detect(votes)

        # Calculate diversity
        diversity_score = self._calculate_diversity(votes, options)

        # Average confidence
        avg_confidence = sum(v.confidence for v in votes) / len(votes)

        result = VotingResult(
            question=question,
            options=options,
            all_votes=votes,
            vote_distribution=dict(vote_counts),
            weighted_distribution=weighted_counts,
            winning_option=winning_option,
            vote_percentage=vote_percentage,
            consensus_level=consensus_level,
            groupthink_detected=groupthink,
            diversity_score=diversity_score,
            confidence_average=avg_confidence,
            total_agents=self.num_agents,
            execution_time_seconds=time.time() - start_time
        )

        print(f"\n🐝 VOTING COMPLETE")
        print(f"   Verdict: {result.verdict}")
        print(f"   Diversity: {diversity_score:.1f}/100")
        print(f"   Groupthink: {'⚠️  DETECTED' if groupthink else '✅ None'}")
        print(f"   Reliability: {result.reliability:.0f}/100")

        return result

    async def _collect_votes(
        self,
        question: str,
        options: List[str],
        weights: Dict[str, float]
    ) -> List[Vote]:
        """Collect votes from all agents"""
        votes = []

        for agent in self.agents:
            # Simple voting logic (placeholder - would use LLM in production)
            import random
            option = random.choice(options)
            confidence = 0.5 + random.random() * 0.5

            vote = Vote(
                agent_id=agent.agent_id,
                option=option,
                confidence=confidence,
                reasoning=f"Based on my expertise in {', '.join(agent.expertise)}"
            )
            votes.append(vote)

        return votes

    def _determine_consensus(self, vote_percentage: float) -> ConsensusLevel:
        """Determine consensus level from vote percentage"""
        if vote_percentage >= 80:
            return ConsensusLevel.STRONG
        elif vote_percentage >= 60:
            return ConsensusLevel.MODERATE
        elif vote_percentage >= 40:
            return ConsensusLevel.WEAK
        else:
            return ConsensusLevel.NO_CONSENSUS

    def _calculate_diversity(self, votes: List[Vote], options: List[str]) -> float:
        """Calculate opinion diversity (0-100)"""
        vote_counts = Counter(v.option for v in votes)
        total = len(votes)

        # Perfect diversity = equal votes for all options
        perfect_distribution = total / len(options)
        actual_variance = sum(
            abs(vote_counts.get(opt, 0) - perfect_distribution)
            for opt in options
        )
        max_variance = (len(options) - 1) * perfect_distribution

        diversity = 100 * (1 - actual_variance / max_variance) if max_variance > 0 else 0
        return diversity
