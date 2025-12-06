"""Weight Calculator"""

from typing import Dict, List, Optional
from swarm_voting.core.models import AgentVoter


class WeightCalculator:
    """Calculates voting weights based on expertise"""

    def calculate_weights(
        self,
        agents: List[AgentVoter],
        question: str,
        context: Optional[str] = None
    ) -> Dict[str, float]:
        """Calculate weights for each agent based on expertise match"""
        weights = {}

        for agent in agents:
            # Base weight
            weight = 1.0

            # Expertise bonus (if question matches expertise)
            if any(exp in question.lower() or (context and exp in context.lower())
                   for exp in agent.expertise):
                weight *= 1.5

            # Historical accuracy bonus
            weight *= (0.5 + agent.past_accuracy)

            weights[agent.agent_id] = weight

        return weights
