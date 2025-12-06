"""
Matchmaker - Pairs agents for matches
"""

import random
from typing import List, Tuple, Any


class Matchmaker:
    """Creates pairings for tournament matches"""

    def __init__(self, seed: int = None):
        if seed:
            random.seed(seed)

    def create_random_pairings(self, agents: List[Any]) -> List[Tuple[Any, Any]]:
        """Create random pairings"""
        shuffled = agents.copy()
        random.shuffle(shuffled)

        pairs = []
        for i in range(0, len(shuffled)-1, 2):
            pairs.append((shuffled[i], shuffled[i+1]))

        return pairs

    def create_elo_pairings(self, agents: List[Any], elo_ratings: dict) -> List[Tuple[Any, Any]]:
        """Pair agents with similar ELO"""
        sorted_agents = sorted(agents, key=lambda a: elo_ratings[a.id].rating, reverse=True)

        pairs = []
        for i in range(0, len(sorted_agents)-1, 2):
            pairs.append((sorted_agents[i], sorted_agents[i+1]))

        return pairs

    def create_bracket(self, agents: List[Any], seed_by_rating: bool = False, elo_ratings: dict = None) -> List[Tuple[Any, Any]]:
        """Create elimination bracket"""
        if seed_by_rating and elo_ratings:
            sorted_agents = sorted(agents, key=lambda a: elo_ratings[a.id].rating, reverse=True)
        else:
            sorted_agents = agents.copy()
            random.shuffle(sorted_agents)

        pairs = []
        for i in range(0, len(sorted_agents)-1, 2):
            pairs.append((sorted_agents[i], sorted_agents[i+1]))

        return pairs
