"""
ELO Rating System
"""

from agent_tournaments.core.models import ELORating, MatchResult


class ELOSystem:
    """ELO rating system for agents"""

    def __init__(self, k_factor: int = 32):
        self.k_factor = k_factor

    def calculate_expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for agent A vs agent B"""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(
        self,
        agent_1_id: str,
        agent_2_id: str,
        match: MatchResult,
        elo_ratings: dict
    ):
        """Update ELO ratings after a match"""
        elo_1 = elo_ratings[agent_1_id]
        elo_2 = elo_ratings[agent_2_id]

        # Calculate expected scores
        expected_1 = self.calculate_expected_score(elo_1.rating, elo_2.rating)
        expected_2 = 1 - expected_1

        # Determine actual scores
        if match.winner_id == agent_1_id:
            actual_1, actual_2 = 1.0, 0.0
        elif match.winner_id == agent_2_id:
            actual_1, actual_2 = 0.0, 1.0
        else:  # Draw
            actual_1, actual_2 = 0.5, 0.5

        # Update ratings
        elo_1.update_rating(expected_1, actual_1, self.k_factor)
        elo_2.update_rating(expected_2, actual_2, self.k_factor)
