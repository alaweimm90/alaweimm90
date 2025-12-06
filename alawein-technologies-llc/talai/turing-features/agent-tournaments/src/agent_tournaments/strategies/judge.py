"""
Tournament Judge

LLM-based judging of agent solutions.
"""

from agent_tournaments.core.models import MatchResult
from datetime import datetime


class TournamentJudge:
    """Judges matches between agents"""

    def __init__(self, orchestrator=None, model: str = "gpt-4"):
        self.orchestrator = orchestrator
        self.model = model

    async def judge_match(
        self,
        agent_1,
        solution_1,
        agent_2,
        solution_2,
        problem
    ) -> MatchResult:
        """
        Judge a match between two agents

        Args:
            agent_1: First agent
            solution_1: Agent 1's solution
            agent_2: Second agent
            solution_2: Agent 2's solution
            problem: The problem being solved

        Returns:
            MatchResult with winner and scores
        """

        # Evaluate solutions (placeholder - implement based on problem type)
        score_1 = self._evaluate_solution(solution_1, problem)
        score_2 = self._evaluate_solution(solution_2, problem)

        # Determine winner
        if score_1 > score_2:
            winner_id = agent_1.id
            margin = score_1 - score_2
            reasoning = f"{agent_1.name} produced better solution ({score_1:.2f} vs {score_2:.2f})"
        elif score_2 > score_1:
            winner_id = agent_2.id
            margin = score_2 - score_1
            reasoning = f"{agent_2.name} produced better solution ({score_2:.2f} vs {score_1:.2f})"
        else:
            winner_id = agent_1.id  # Tie goes to agent 1
            margin = 0.0
            reasoning = "Tie - both solutions equal quality"

        return MatchResult(
            match_id=f"match_{agent_1.id}_vs_{agent_2.id}_{int(datetime.now().timestamp())}",
            agent_1_id=agent_1.id,
            agent_2_id=agent_2.id,
            agent_1_solution=solution_1,
            agent_2_solution=solution_2,
            agent_1_score=score_1,
            agent_2_score=score_2,
            winner_id=winner_id,
            margin=margin,
            judge_reasoning=reasoning
        )

    def _evaluate_solution(self, solution, problem) -> float:
        """Evaluate solution quality"""
        # Placeholder - implement based on problem type
        # For optimization: return objective value
        # For other problems: use LLM judgment
        if hasattr(solution, 'objective_value'):
            return solution.objective_value
        return 0.0
