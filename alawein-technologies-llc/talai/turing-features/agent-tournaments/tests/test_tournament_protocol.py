"""
Comprehensive tests for Tournament Protocol.

Tests all tournament formats, ELO system, and edge cases.
"""

import pytest
from typing import List, Any
from unittest.mock import Mock, AsyncMock, patch

from agent_tournaments import (
    TournamentProtocol,
    TournamentFormat,
    TournamentResult,
    MatchResult,
    AgentRanking,
    ELORating,
)
from agent_tournaments.core.models import Agent, Problem


# Fixtures


@pytest.fixture
def mock_orchestrator():
    """Mock TalAI orchestrator."""
    orchestrator = Mock()
    orchestrator.call_llm = AsyncMock(
        return_value="Agent A wins with superior solution quality."
    )
    return orchestrator


@pytest.fixture
def sample_agents() -> List[Agent]:
    """Create sample agents for testing."""
    return [
        Agent(agent_id=f"agent_{i}", agent_name=f"Agent {i}", expertise=["optimization"])
        for i in range(8)
    ]


@pytest.fixture
def sample_problem() -> Problem:
    """Create sample problem."""
    return Problem(
        description="Optimize a quadratic function",
        domain="optimization",
        difficulty="medium",
    )


@pytest.fixture
def tournament_protocol(mock_orchestrator):
    """Create tournament protocol with mocked orchestrator."""
    return TournamentProtocol(orchestrator=mock_orchestrator)


# Test: Initialization


class TestTournamentProtocol:
    """Test suite for TournamentProtocol."""

    def test_initialization(self, mock_orchestrator):
        """Test protocol initialization."""
        protocol = TournamentProtocol(orchestrator=mock_orchestrator)
        assert protocol.orchestrator == mock_orchestrator
        assert protocol.k_factor == 32

    def test_initialization_custom_k_factor(self, mock_orchestrator):
        """Test initialization with custom K-factor."""
        protocol = TournamentProtocol(orchestrator=mock_orchestrator, k_factor=16)
        assert protocol.k_factor == 16


# Test: Elimination Tournament


class TestEliminationTournament:
    """Test elimination tournament format."""

    @pytest.mark.asyncio
    async def test_elimination_basic(
        self, tournament_protocol, sample_agents, sample_problem
    ):
        """Test basic elimination tournament."""
        result = await tournament_protocol.run_tournament(
            agents=sample_agents, problem=sample_problem, format=TournamentFormat.ELIMINATION
        )

        assert isinstance(result, TournamentResult)
        assert result.tournament_format == TournamentFormat.ELIMINATION
        assert result.champion is not None
        assert result.champion.agent_name.startswith("Agent")
        assert len(result.rankings) == len(sample_agents)

    @pytest.mark.asyncio
    async def test_elimination_with_odd_agents(self, tournament_protocol, sample_problem):
        """Test elimination with odd number of agents (requires bye)."""
        agents = [
            Agent(agent_id=f"agent_{i}", agent_name=f"Agent {i}")
            for i in range(7)  # Odd number
        ]

        result = await tournament_protocol.run_tournament(
            agents=agents, problem=sample_problem, format=TournamentFormat.ELIMINATION
        )

        assert result.champion is not None
        assert len(result.rankings) == 7

    @pytest.mark.asyncio
    async def test_elimination_bracket_structure(
        self, tournament_protocol, sample_agents, sample_problem
    ):
        """Test that elimination creates proper bracket."""
        result = await tournament_protocol.run_tournament(
            agents=sample_agents, problem=sample_problem, format=TournamentFormat.ELIMINATION
        )

        # With 8 agents: Round 1 (4 matches), Round 2 (2 matches), Round 3 (1 match)
        # Total: 7 matches
        assert len(result.matches) == 7


# Test: Round-Robin Tournament


class TestRoundRobinTournament:
    """Test round-robin tournament format."""

    @pytest.mark.asyncio
    async def test_round_robin_basic(
        self, tournament_protocol, sample_agents, sample_problem
    ):
        """Test basic round-robin tournament."""
        result = await tournament_protocol.run_tournament(
            agents=sample_agents, problem=sample_problem, format=TournamentFormat.ROUND_ROBIN
        )

        assert result.tournament_format == TournamentFormat.ROUND_ROBIN
        assert result.champion is not None

    @pytest.mark.asyncio
    async def test_round_robin_all_pairings(
        self, tournament_protocol, sample_agents, sample_problem
    ):
        """Test that round-robin creates all possible pairings."""
        n = len(sample_agents)
        expected_matches = n * (n - 1) // 2  # n choose 2

        result = await tournament_protocol.run_tournament(
            agents=sample_agents, problem=sample_problem, format=TournamentFormat.ROUND_ROBIN
        )

        assert len(result.matches) == expected_matches

    @pytest.mark.asyncio
    async def test_round_robin_fairness(
        self, tournament_protocol, sample_agents, sample_problem
    ):
        """Test that every agent plays every other agent exactly once."""
        result = await tournament_protocol.run_tournament(
            agents=sample_agents, problem=sample_problem, format=TournamentFormat.ROUND_ROBIN
        )

        # Track pairings
        pairings = set()
        for match in result.matches:
            pair = tuple(sorted([match.agent_a_id, match.agent_b_id]))
            assert pair not in pairings, "Duplicate pairing found!"
            pairings.add(pair)


# Test: Swiss Tournament


class TestSwissTournament:
    """Test Swiss system tournament format."""

    @pytest.mark.asyncio
    async def test_swiss_basic(self, tournament_protocol, sample_agents, sample_problem):
        """Test basic Swiss tournament."""
        result = await tournament_protocol.run_tournament(
            agents=sample_agents, problem=sample_problem, format=TournamentFormat.SWISS
        )

        assert result.tournament_format == TournamentFormat.SWISS
        assert result.champion is not None

    @pytest.mark.asyncio
    async def test_swiss_multiple_rounds(
        self, tournament_protocol, sample_agents, sample_problem
    ):
        """Test that Swiss runs correct number of rounds."""
        result = await tournament_protocol.run_tournament(
            agents=sample_agents, problem=sample_problem, format=TournamentFormat.SWISS
        )

        # Swiss typically runs n/2 rounds for n agents
        expected_rounds = len(sample_agents) // 2
        # Each round has n/2 matches
        expected_matches = expected_rounds * (len(sample_agents) // 2)

        assert len(result.matches) >= expected_matches - 4  # Allow some tolerance


# Test: Free-for-All Tournament


class TestFreeForAllTournament:
    """Test free-for-all tournament format."""

    @pytest.mark.asyncio
    async def test_free_for_all_basic(
        self, tournament_protocol, sample_agents, sample_problem
    ):
        """Test basic free-for-all tournament."""
        result = await tournament_protocol.run_tournament(
            agents=sample_agents, problem=sample_problem, format=TournamentFormat.FREE_FOR_ALL
        )

        assert result.tournament_format == TournamentFormat.FREE_FOR_ALL
        assert result.champion is not None
        assert len(result.rankings) == len(sample_agents)

    @pytest.mark.asyncio
    async def test_free_for_all_single_round(
        self, tournament_protocol, sample_agents, sample_problem
    ):
        """Test that free-for-all completes in single evaluation round."""
        result = await tournament_protocol.run_tournament(
            agents=sample_agents, problem=sample_problem, format=TournamentFormat.FREE_FOR_ALL
        )

        # Free-for-all should have minimal matches (just rankings)
        assert len(result.matches) <= len(sample_agents)


# Test: Multi-Stage Tournament


class TestMultiStageTournament:
    """Test multi-stage tournament format."""

    @pytest.mark.asyncio
    async def test_multi_stage_basic(self, tournament_protocol, sample_problem):
        """Test basic multi-stage tournament."""
        # Create larger agent pool for multi-stage
        agents = [Agent(agent_id=f"agent_{i}", agent_name=f"Agent {i}") for i in range(32)]

        result = await tournament_protocol.run_tournament(
            agents=agents, problem=sample_problem, format=TournamentFormat.MULTI_STAGE
        )

        assert result.tournament_format == TournamentFormat.MULTI_STAGE
        assert result.champion is not None

    @pytest.mark.asyncio
    async def test_multi_stage_progressive_filtering(
        self, tournament_protocol, sample_problem
    ):
        """Test that multi-stage progressively filters agents."""
        agents = [Agent(agent_id=f"agent_{i}", agent_name=f"Agent {i}") for i in range(100)]

        result = await tournament_protocol.run_tournament(
            agents=agents, problem=sample_problem, format=TournamentFormat.MULTI_STAGE
        )

        # Should have champion from large pool
        assert result.champion is not None
        assert len(result.rankings) == 100


# Test: ELO Rating System


class TestELORatingSystem:
    """Test ELO rating calculations."""

    def test_elo_expected_score_equal_ratings(self, tournament_protocol):
        """Test expected score with equal ratings."""
        from agent_tournaments.strategies.elo_system import ELOSystem

        elo = ELOSystem(k_factor=32)
        expected = elo.calculate_expected_score(1500, 1500)
        assert abs(expected - 0.5) < 0.01  # Should be 50%

    def test_elo_expected_score_higher_rating(self, tournament_protocol):
        """Test expected score with higher rating."""
        from agent_tournaments.strategies.elo_system import ELOSystem

        elo = ELOSystem(k_factor=32)
        expected = elo.calculate_expected_score(1700, 1500)
        assert expected > 0.75  # Should be >75%

    def test_elo_rating_update_win(self, tournament_protocol):
        """Test ELO update after win."""
        from agent_tournaments.strategies.elo_system import ELOSystem

        elo = ELOSystem(k_factor=32)
        rating_a = ELORating(agent_id="a", current_rating=1500)
        rating_b = ELORating(agent_id="b", current_rating=1500)

        updated_a, updated_b = elo.update_ratings(rating_a, rating_b, winner="a")

        assert updated_a.current_rating > 1500  # Winner gains rating
        assert updated_b.current_rating < 1500  # Loser loses rating
        assert updated_a.wins == 1
        assert updated_b.losses == 1

    def test_elo_rating_update_draw(self, tournament_protocol):
        """Test ELO update after draw."""
        from agent_tournaments.strategies.elo_system import ELOSystem

        elo = ELOSystem(k_factor=32)
        rating_a = ELORating(agent_id="a", current_rating=1500)
        rating_b = ELORating(agent_id="b", current_rating=1500)

        updated_a, updated_b = elo.update_ratings(rating_a, rating_b, winner=None)

        assert updated_a.current_rating == 1500  # No change in equal draw
        assert updated_b.current_rating == 1500
        assert updated_a.draws == 1
        assert updated_b.draws == 1


# Test: Competitive Pressure Effect


class TestCompetitivePressure:
    """Test competitive pressure metrics."""

    @pytest.mark.asyncio
    async def test_competitive_pressure_calculated(
        self, tournament_protocol, sample_agents, sample_problem
    ):
        """Test that competitive pressure effect is calculated."""
        result = await tournament_protocol.run_tournament(
            agents=sample_agents, problem=sample_problem, format=TournamentFormat.ELIMINATION
        )

        assert result.competitive_pressure_effect >= 0  # Should show improvement

    @pytest.mark.asyncio
    async def test_improvement_over_baseline(
        self, tournament_protocol, sample_agents, sample_problem
    ):
        """Test improvement calculation with baseline."""
        baseline_solution = Mock(quality=50.0)

        result = await tournament_protocol.run_tournament(
            agents=sample_agents,
            problem=sample_problem,
            format=TournamentFormat.ELIMINATION,
            baseline_solution=baseline_solution,
        )

        assert hasattr(result, "improvement_over_baseline")


# Test: Edge Cases


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_single_agent(self, tournament_protocol, sample_problem):
        """Test tournament with single agent."""
        agents = [Agent(agent_id="solo", agent_name="Solo Agent")]

        result = await tournament_protocol.run_tournament(
            agents=agents, problem=sample_problem, format=TournamentFormat.FREE_FOR_ALL
        )

        assert result.champion.agent_id == "solo"
        assert len(result.rankings) == 1

    @pytest.mark.asyncio
    async def test_two_agents(self, tournament_protocol, sample_problem):
        """Test tournament with exactly two agents."""
        agents = [
            Agent(agent_id="agent_0", agent_name="Agent 0"),
            Agent(agent_id="agent_1", agent_name="Agent 1"),
        ]

        result = await tournament_protocol.run_tournament(
            agents=agents, problem=sample_problem, format=TournamentFormat.ELIMINATION
        )

        assert result.champion is not None
        assert len(result.rankings) == 2

    @pytest.mark.asyncio
    async def test_empty_agents_raises_error(self, tournament_protocol, sample_problem):
        """Test that empty agent list raises error."""
        with pytest.raises((ValueError, AssertionError)):
            await tournament_protocol.run_tournament(
                agents=[], problem=sample_problem, format=TournamentFormat.ELIMINATION
            )


# Test: Rankings


class TestRankings:
    """Test ranking generation."""

    @pytest.mark.asyncio
    async def test_rankings_complete(
        self, tournament_protocol, sample_agents, sample_problem
    ):
        """Test that all agents are ranked."""
        result = await tournament_protocol.run_tournament(
            agents=sample_agents, problem=sample_problem, format=TournamentFormat.ROUND_ROBIN
        )

        assert len(result.rankings) == len(sample_agents)

    @pytest.mark.asyncio
    async def test_rankings_ordered(
        self, tournament_protocol, sample_agents, sample_problem
    ):
        """Test that rankings are properly ordered."""
        result = await tournament_protocol.run_tournament(
            agents=sample_agents, problem=sample_problem, format=TournamentFormat.ROUND_ROBIN
        )

        # Check rankings are in descending ELO order
        elos = [r.elo_rating for r in result.rankings]
        assert elos == sorted(elos, reverse=True)

    @pytest.mark.asyncio
    async def test_champion_is_top_ranked(
        self, tournament_protocol, sample_agents, sample_problem
    ):
        """Test that champion matches top ranking."""
        result = await tournament_protocol.run_tournament(
            agents=sample_agents, problem=sample_problem, format=TournamentFormat.ELIMINATION
        )

        assert result.champion.agent_id == result.rankings[0].agent_id
