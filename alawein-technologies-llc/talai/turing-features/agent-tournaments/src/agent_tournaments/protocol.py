"""
Tournament Protocol

Main orchestrator for agent tournaments.
"""

import time
import asyncio
from typing import List, Optional, Any
from pathlib import Path

from agent_tournaments.core.models import (
    TournamentFormat,
    TournamentResult,
    TournamentConfig,
    AgentRanking,
    MatchResult,
    ELORating,
)
from agent_tournaments.core.formats import get_tournament_class
from agent_tournaments.strategies.elo_system import ELOSystem
from agent_tournaments.strategies.judge import TournamentJudge


class TournamentProtocol:
    """
    Agent Tournament System

    Competitive selection through tournaments with 5 formats:
    - Free-for-all: All agents compete simultaneously
    - Elimination: Bracket-style pairwise elimination
    - Round-robin: Every agent plays every other agent
    - Swiss: Multiple rounds pairing similar performers
    - Multi-stage: Combination of formats

    Usage:
        protocol = TournamentProtocol()
        result = await protocol.run_tournament(
            agents=agents,
            problem=problem,
            format=TournamentFormat.ELIMINATION
        )
    """

    def __init__(self, orchestrator=None, **config):
        """
        Initialize tournament protocol

        Args:
            orchestrator: AI Orchestrator for LLM calls
            **config: Tournament configuration
        """
        self.orchestrator = orchestrator
        self.config = TournamentConfig(**config)

        # Initialize components
        self.elo_system = ELOSystem(k_factor=self.config.k_factor)
        self.judge = TournamentJudge(
            orchestrator=orchestrator,
            model=self.config.judge_model
        )

    async def run_tournament(
        self,
        agents: List[Any],
        problem: Any,
        format: TournamentFormat = TournamentFormat.ELIMINATION,
        baseline_solution: Optional[Any] = None
    ) -> TournamentResult:
        """
        Run a tournament with given agents and problem

        Args:
            agents: List of agents to compete
            problem: Problem to solve
            format: Tournament format to use
            baseline_solution: Baseline to compare against

        Returns:
            TournamentResult with rankings and analysis
        """
        start_time = time.time()

        print(f"\n🏆 TOURNAMENT START: {format.value}")
        print(f"   Agents: {len(agents)}")
        print(f"   Format: {format.value}")

        # Initialize ELO ratings for all agents
        elo_ratings = {
            agent.id: ELORating(agent_id=agent.id)
            for agent in agents
        }

        # Get tournament class and run
        tournament_class = get_tournament_class(format)
        tournament = tournament_class(
            agents=agents,
            problem=problem,
            judge=self.judge,
            elo_system=self.elo_system,
            elo_ratings=elo_ratings,
            config=self.config
        )

        # Execute tournament
        matches, rounds = await tournament.execute()

        # Generate final rankings
        rankings = self._generate_rankings(agents, elo_ratings, matches)

        # Calculate statistics
        champion = rankings[0]
        champion_solution = matches[-1].agent_1_solution if matches[-1].winner_id == champion.agent_id else matches[-1].agent_2_solution

        avg_quality = sum(r.total_score for r in rankings) / len(rankings)

        # Calculate improvements
        improvement_baseline = 0.0
        if baseline_solution:
            baseline_quality = self._evaluate_solution(baseline_solution)
            improvement_baseline = ((champion.best_solution_quality - baseline_quality) / baseline_quality * 100)

        # Build result
        result = TournamentResult(
            tournament_id=f"tournament_{int(time.time())}",
            format=format,
            problem=problem,
            rankings=rankings,
            all_matches=matches,
            champion=champion,
            champion_solution=champion_solution,
            total_agents=len(agents),
            total_matches=len(matches),
            total_rounds=rounds,
            average_solution_quality=avg_quality,
            improvement_over_random=champion.best_solution_quality - avg_quality,
            improvement_over_baseline=improvement_baseline,
            execution_time_seconds=time.time() - start_time,
            models_used=[self.config.judge_model]
        )

        print(f"\n🏁 TOURNAMENT COMPLETE")
        print(f"   Champion: {champion.agent_name}")
        print(f"   Quality: {champion.best_solution_quality:.2f}")
        print(f"   Matches: {len(matches)}")
        print(f"   Time: {result.execution_time_seconds:.1f}s")
        print(f"   Competitive improvement: {result.competitive_pressure_effect:.1f}%")

        return result

    def _generate_rankings(
        self,
        agents: List[Any],
        elo_ratings: dict,
        matches: List[MatchResult]
    ) -> List[AgentRanking]:
        """Generate final rankings from ELO ratings and match results"""
        rankings = []

        for agent in agents:
            elo = elo_ratings[agent.id]

            # Calculate best solution quality from matches
            agent_matches = [m for m in matches if m.agent_1_id == agent.id or m.agent_2_id == agent.id]
            best_quality = max(
                (m.agent_1_score if m.agent_1_id == agent.id else m.agent_2_score)
                for m in agent_matches
            ) if agent_matches else 0.0

            # Calculate total score
            total_score = sum(
                (m.agent_1_score if m.agent_1_id == agent.id else m.agent_2_score)
                for m in agent_matches
            )

            # Calculate consistency (variance of scores)
            scores = [(m.agent_1_score if m.agent_1_id == agent.id else m.agent_2_score) for m in agent_matches]
            consistency = 100 - (max(scores) - min(scores)) / max(scores) * 100 if scores else 0.0

            ranking = AgentRanking(
                rank=0,  # Will be assigned after sorting
                agent_id=agent.id,
                agent_name=agent.name,
                elo_rating=elo.rating,
                wins=elo.wins,
                losses=elo.losses,
                draws=elo.draws,
                total_score=total_score,
                best_solution_quality=best_quality,
                consistency_score=consistency
            )
            rankings.append(ranking)

        # Sort by ELO rating and assign ranks
        rankings.sort(key=lambda r: r.elo_rating, reverse=True)
        for i, ranking in enumerate(rankings, 1):
            ranking.rank = i

        return rankings

    def _evaluate_solution(self, solution: Any) -> float:
        """Evaluate a solution quality"""
        # Placeholder - implement based on problem type
        return 0.0

    async def run_tournament_batch(
        self,
        agents: List[Any],
        problems: List[Any],
        format: TournamentFormat = TournamentFormat.ELIMINATION
    ) -> List[TournamentResult]:
        """Run tournaments on multiple problems"""
        results = []
        for problem in problems:
            result = await self.run_tournament(agents, problem, format)
            results.append(result)
        return results
