"""
Tournament Format Implementations
"""

import asyncio
import random
from typing import List, Tuple, Any
from abc import ABC, abstractmethod

from agent_tournaments.core.models import MatchResult, TournamentFormat


class BaseTournamentFormat(ABC):
    """Base class for tournament formats"""

    def __init__(self, agents, problem, judge, elo_system, elo_ratings, config):
        self.agents = agents
        self.problem = problem
        self.judge = judge
        self.elo_system = elo_system
        self.elo_ratings = elo_ratings
        self.config = config

    @abstractmethod
    async def execute(self) -> Tuple[List[MatchResult], int]:
        """Execute tournament, return (matches, rounds)"""
        pass


class FreeForAllTournament(BaseTournamentFormat):
    """All agents compete simultaneously"""

    async def execute(self) -> Tuple[List[MatchResult], int]:
        # All agents solve simultaneously
        solutions = {}
        for agent in self.agents:
            solution = await agent.solve(self.problem)
            solutions[agent.id] = solution

        # Judge each pair
        matches = []
        for i, agent1 in enumerate(self.agents):
            for agent2 in self.agents[i+1:]:
                match = await self.judge.judge_match(
                    agent1, solutions[agent1.id],
                    agent2, solutions[agent2.id],
                    self.problem
                )
                self.elo_system.update_ratings(agent1.id, agent2.id, match, self.elo_ratings)
                matches.append(match)

        return matches, 1


class EliminationTournament(BaseTournamentFormat):
    """Bracket-style elimination"""

    async def execute(self) -> Tuple[List[MatchResult], int]:
        current_agents = self.agents.copy()
        all_matches = []
        round_num = 0

        while len(current_agents) > 1:
            round_num += 1
            print(f"   Round {round_num}: {len(current_agents)} agents")

            round_winners = []
            round_matches = []

            # Pair agents and run matches
            for i in range(0, len(current_agents), 2):
                if i + 1 < len(current_agents):
                    agent1 = current_agents[i]
                    agent2 = current_agents[i + 1]

                    sol1 = await agent1.solve(self.problem)
                    sol2 = await agent2.solve(self.problem)

                    match = await self.judge.judge_match(
                        agent1, sol1, agent2, sol2, self.problem
                    )

                    self.elo_system.update_ratings(agent1.id, agent2.id, match, self.elo_ratings)
                    round_matches.append(match)

                    winner = agent1 if match.winner_id == agent1.id else agent2
                    round_winners.append(winner)
                else:
                    # Bye round
                    round_winners.append(current_agents[i])

            all_matches.extend(round_matches)
            current_agents = round_winners

        return all_matches, round_num


class RoundRobinTournament(BaseTournamentFormat):
    """Every agent plays every other agent"""

    async def execute(self) -> Tuple[List[MatchResult], int]:
        matches = []

        for i, agent1 in enumerate(self.agents):
            for agent2 in self.agents[i+1:]:
                sol1 = await agent1.solve(self.problem)
                sol2 = await agent2.solve(self.problem)

                match = await self.judge.judge_match(
                    agent1, sol1, agent2, sol2, self.problem
                )

                self.elo_system.update_ratings(agent1.id, agent2.id, match, self.elo_ratings)
                matches.append(match)

        return matches, 1


class SwissTournament(BaseTournamentFormat):
    """Pair similar performers"""

    async def execute(self) -> Tuple[List[MatchResult], int]:
        num_rounds = len(self.agents) // 2
        all_matches = []

        for round_num in range(num_rounds):
            print(f"   Swiss Round {round_num + 1}")

            # Sort agents by ELO
            sorted_agents = sorted(
                self.agents,
                key=lambda a: self.elo_ratings[a.id].rating,
                reverse=True
            )

            # Pair agents
            round_matches = []
            for i in range(0, len(sorted_agents), 2):
                if i + 1 < len(sorted_agents):
                    agent1 = sorted_agents[i]
                    agent2 = sorted_agents[i + 1]

                    sol1 = await agent1.solve(self.problem)
                    sol2 = await agent2.solve(self.problem)

                    match = await self.judge.judge_match(
                        agent1, sol1, agent2, sol2, self.problem
                    )

                    self.elo_system.update_ratings(agent1.id, agent2.id, match, self.elo_ratings)
                    round_matches.append(match)

            all_matches.extend(round_matches)

        return all_matches, num_rounds


class MultiStageTournament(BaseTournamentFormat):
    """Combination of formats"""

    async def execute(self) -> Tuple[List[MatchResult], int]:
        # Stage 1: Free-for-all (filter to top 50%)
        print("   Stage 1: Free-for-all qualification")
        ffa = FreeForAllTournament(
            self.agents, self.problem, self.judge,
            self.elo_system, self.elo_ratings, self.config
        )
        matches1, _ = await ffa.execute()

        # Get top 50%
        sorted_agents = sorted(
            self.agents,
            key=lambda a: self.elo_ratings[a.id].rating,
            reverse=True
        )
        semifinalists = sorted_agents[:len(sorted_agents)//2]

        # Stage 2: Round-robin among survivors
        print(f"   Stage 2: Round-robin ({len(semifinalists)} agents)")
        rr = RoundRobinTournament(
            semifinalists, self.problem, self.judge,
            self.elo_system, self.elo_ratings, self.config
        )
        matches2, _ = await rr.execute()

        # Stage 3: Elimination for top 4
        finalists = sorted(
            semifinalists,
            key=lambda a: self.elo_ratings[a.id].rating,
            reverse=True
        )[:4]

        print(f"   Stage 3: Elimination finals ({len(finalists)} agents)")
        elim = EliminationTournament(
            finalists, self.problem, self.judge,
            self.elo_system, self.elo_ratings, self.config
        )
        matches3, rounds = await elim.execute()

        return matches1 + matches2 + matches3, 3


def get_tournament_class(format: TournamentFormat):
    """Get tournament class for format"""
    formats = {
        TournamentFormat.FREE_FOR_ALL: FreeForAllTournament,
        TournamentFormat.ELIMINATION: EliminationTournament,
        TournamentFormat.ROUND_ROBIN: RoundRobinTournament,
        TournamentFormat.SWISS: SwissTournament,
        TournamentFormat.MULTI_STAGE: MultiStageTournament,
    }
    return formats[format]
