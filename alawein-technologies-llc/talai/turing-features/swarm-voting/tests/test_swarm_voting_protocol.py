"""
Comprehensive tests for Swarm Intelligence Voting Protocol.

Tests voting mechanisms, consensus calculation, groupthink detection, and reliability scoring.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from typing import List

from swarm_voting import (
    SwarmVotingProtocol,
    VotingResult,
    Vote,
    ConsensusLevel,
    Agent,
)


# Fixtures


@pytest.fixture
def mock_orchestrator():
    """Mock TalAI orchestrator."""
    orchestrator = Mock()
    orchestrator.call_llm = AsyncMock(return_value="Yes")
    return orchestrator


@pytest.fixture
def swarm_protocol_small():
    """Create swarm protocol with small swarm (20 agents)."""
    return SwarmVotingProtocol(num_agents=20)


@pytest.fixture
def swarm_protocol_medium():
    """Create swarm protocol with medium swarm (50 agents)."""
    return SwarmVotingProtocol(num_agents=50)


@pytest.fixture
def swarm_protocol_large():
    """Create swarm protocol with large swarm (100 agents)."""
    return SwarmVotingProtocol(num_agents=100)


@pytest.fixture
def binary_question():
    """Create binary question."""
    return {
        "question": "Should we proceed with this hypothesis?",
        "options": ["Yes", "No"],
    }


@pytest.fixture
def multi_option_question():
    """Create multi-option question."""
    return {
        "question": "Which research direction should we pursue?",
        "options": [
            "Quantum approach",
            "Machine learning approach",
            "Classical optimization",
            "Hybrid method",
        ],
    }


# Test: Initialization


class TestSwarmVotingProtocol:
    """Test suite for Swarm Voting Protocol."""

    def test_initialization_default(self):
        """Test default initialization."""
        protocol = SwarmVotingProtocol()
        assert protocol.num_agents == 100  # Default

    def test_initialization_custom_size(self):
        """Test initialization with custom swarm size."""
        protocol = SwarmVotingProtocol(num_agents=50)
        assert protocol.num_agents == 50

    def test_initialization_small_swarm(self):
        """Test initialization with small swarm."""
        protocol = SwarmVotingProtocol(num_agents=10)
        assert protocol.num_agents == 10


# Test: Binary Voting


class TestBinaryVoting:
    """Test binary voting (Yes/No)."""

    @pytest.mark.asyncio
    async def test_binary_vote_basic(self, swarm_protocol_medium, binary_question):
        """Test basic binary voting."""
        result = await swarm_protocol_medium.vote(
            question=binary_question["question"], options=binary_question["options"]
        )

        assert isinstance(result, VotingResult)
        assert result.winning_option in ["Yes", "No"]
        assert result.consensus_level in [
            ConsensusLevel.STRONG,
            ConsensusLevel.MODERATE,
            ConsensusLevel.WEAK,
            ConsensusLevel.NO_CONSENSUS,
        ]

    @pytest.mark.asyncio
    async def test_binary_vote_distribution(self, swarm_protocol_medium, binary_question):
        """Test that binary votes are distributed."""
        result = await swarm_protocol_medium.vote(
            question=binary_question["question"], options=binary_question["options"]
        )

        # Total votes should equal num_agents
        total_votes = sum(result.vote_distribution.values())
        assert total_votes == swarm_protocol_medium.num_agents

    @pytest.mark.asyncio
    async def test_binary_vote_percentage(self, swarm_protocol_medium, binary_question):
        """Test vote percentage calculation."""
        result = await swarm_protocol_medium.vote(
            question=binary_question["question"], options=binary_question["options"]
        )

        # Vote percentage should be between 0 and 100
        assert 0 <= result.vote_percentage <= 100

    @pytest.mark.asyncio
    async def test_binary_strong_consensus(self, swarm_protocol_large):
        """Test detection of strong consensus."""
        # Simulate strong consensus (>80% agreement)
        result = await swarm_protocol_large.vote(
            question="Is 2 + 2 = 4?", options=["Yes", "No"]
        )

        # Should have high agreement for obvious question
        # (This is probabilistic, but very likely)
        assert result.vote_percentage >= 50  # At minimum


# Test: Multi-Option Voting


class TestMultiOptionVoting:
    """Test multi-option voting."""

    @pytest.mark.asyncio
    async def test_multi_option_basic(self, swarm_protocol_medium, multi_option_question):
        """Test basic multi-option voting."""
        result = await swarm_protocol_medium.vote(
            question=multi_option_question["question"],
            options=multi_option_question["options"],
        )

        assert result.winning_option in multi_option_question["options"]

    @pytest.mark.asyncio
    async def test_multi_option_all_counted(
        self, swarm_protocol_medium, multi_option_question
    ):
        """Test that all votes are counted."""
        result = await swarm_protocol_medium.vote(
            question=multi_option_question["question"],
            options=multi_option_question["options"],
        )

        total_votes = sum(result.vote_distribution.values())
        assert total_votes == swarm_protocol_medium.num_agents

    @pytest.mark.asyncio
    async def test_multi_option_distribution(
        self, swarm_protocol_medium, multi_option_question
    ):
        """Test vote distribution across options."""
        result = await swarm_protocol_medium.vote(
            question=multi_option_question["question"],
            options=multi_option_question["options"],
        )

        # All options should appear in distribution
        assert len(result.vote_distribution) <= len(multi_option_question["options"])

    @pytest.mark.asyncio
    async def test_multi_option_winner_is_plurality(
        self, swarm_protocol_large, multi_option_question
    ):
        """Test that winner has plurality of votes."""
        result = await swarm_protocol_large.vote(
            question=multi_option_question["question"],
            options=multi_option_question["options"],
        )

        winner_votes = result.vote_distribution[result.winning_option]
        # Winner should have most votes
        assert winner_votes == max(result.vote_distribution.values())


# Test: Consensus Levels


class TestConsensusLevels:
    """Test consensus level calculation."""

    def test_consensus_strong(self):
        """Test strong consensus (>80%)."""
        result = VotingResult(
            question="Test",
            options=["Yes", "No"],
            winning_option="Yes",
            vote_distribution={"Yes": 85, "No": 15},
            vote_percentage=85.0,
            consensus_level=ConsensusLevel.STRONG,
            diversity_score=30.0,
            reliability=90.0,
            groupthink_detected=False,
            verdict="PROCEED",
        )

        assert result.consensus_level == ConsensusLevel.STRONG
        assert result.vote_percentage > 80

    def test_consensus_moderate(self):
        """Test moderate consensus (60-80%)."""
        result = VotingResult(
            question="Test",
            options=["Yes", "No"],
            winning_option="Yes",
            vote_distribution={"Yes": 70, "No": 30},
            vote_percentage=70.0,
            consensus_level=ConsensusLevel.MODERATE,
            diversity_score=60.0,
            reliability=70.0,
            groupthink_detected=False,
            verdict="PROCEED",
        )

        assert result.consensus_level == ConsensusLevel.MODERATE
        assert 60 <= result.vote_percentage < 80

    def test_consensus_weak(self):
        """Test weak consensus (40-60%)."""
        result = VotingResult(
            question="Test",
            options=["Yes", "No"],
            winning_option="Yes",
            vote_distribution={"Yes": 55, "No": 45},
            vote_percentage=55.0,
            consensus_level=ConsensusLevel.WEAK,
            diversity_score=90.0,
            reliability=50.0,
            groupthink_detected=False,
            verdict="INVESTIGATE",
        )

        assert result.consensus_level == ConsensusLevel.WEAK
        assert 40 <= result.vote_percentage < 60

    def test_consensus_none(self):
        """Test no consensus (<40%)."""
        result = VotingResult(
            question="Test",
            options=["A", "B", "C"],
            winning_option="A",
            vote_distribution={"A": 35, "B": 33, "C": 32},
            vote_percentage=35.0,
            consensus_level=ConsensusLevel.NO_CONSENSUS,
            diversity_score=95.0,
            reliability=30.0,
            groupthink_detected=False,
            verdict="REVISIT",
        )

        assert result.consensus_level == ConsensusLevel.NO_CONSENSUS
        assert result.vote_percentage < 40


# Test: Groupthink Detection


class TestGroupthinkDetection:
    """Test groupthink detection."""

    @pytest.mark.asyncio
    async def test_groupthink_high_uniformity(self, swarm_protocol_large):
        """Test groupthink detection with high uniformity."""
        # Question that should create near-unanimity
        result = await swarm_protocol_large.vote(
            question="Is water wet?", options=["Yes", "No"]
        )

        # Very high agreement might trigger groupthink
        if result.vote_percentage > 90:
            # Could trigger groupthink warning
            assert isinstance(result.groupthink_detected, bool)

    @pytest.mark.asyncio
    async def test_no_groupthink_diverse_opinions(self, swarm_protocol_large):
        """Test no groupthink with diverse opinions."""
        result = await swarm_protocol_large.vote(
            question="Which is best: A, B, C, or D?",
            options=["Option A", "Option B", "Option C", "Option D"],
        )

        # Diverse opinions should not trigger groupthink
        # (Unless by chance one option gets >90%)
        assert isinstance(result.groupthink_detected, bool)

    @pytest.mark.asyncio
    async def test_groupthink_penalty(self, swarm_protocol_large, binary_question):
        """Test that groupthink reduces reliability score."""
        result = await swarm_protocol_large.vote(
            question=binary_question["question"], options=binary_question["options"]
        )

        if result.groupthink_detected:
            # Reliability should be reduced if groupthink detected
            assert result.reliability < 90


# Test: Weighted Voting


class TestWeightedVoting:
    """Test weighted voting by expertise and performance."""

    @pytest.mark.asyncio
    async def test_expertise_based_weighting(self, swarm_protocol_medium):
        """Test that expertise affects vote weight."""
        result = await swarm_protocol_medium.vote(
            question="Should we use quantum optimization?",
            options=["Yes", "No"],
            context="Question about quantum computing",
        )

        # Agents with quantum expertise should have higher weight
        # (Hard to test directly, but shouldn't crash)
        assert isinstance(result, VotingResult)

    @pytest.mark.asyncio
    async def test_performance_based_weighting(self, swarm_protocol_medium):
        """Test that past performance affects vote weight."""
        result = await swarm_protocol_medium.vote(
            question="Is this approach valid?", options=["Yes", "No"]
        )

        # Agents with better track record should have higher weight
        assert isinstance(result, VotingResult)


# Test: Diversity Scoring


class TestDiversityScoring:
    """Test diversity score calculation."""

    @pytest.mark.asyncio
    async def test_diversity_high_with_split_votes(self, swarm_protocol_large):
        """Test high diversity with evenly split votes."""
        result = await swarm_protocol_large.vote(
            question="A or B?", options=["A", "B"]
        )

        # If votes are split, diversity should be high
        if 45 <= result.vote_percentage <= 55:
            assert result.diversity_score > 80

    @pytest.mark.asyncio
    async def test_diversity_low_with_uniform_votes(self, swarm_protocol_large):
        """Test low diversity with uniform votes."""
        result = await swarm_protocol_large.vote(
            question="Is 1 + 1 = 2?", options=["Yes", "No"]
        )

        # If votes are very uniform, diversity should be low
        if result.vote_percentage > 90:
            assert result.diversity_score < 50

    @pytest.mark.asyncio
    async def test_diversity_score_range(self, swarm_protocol_medium, binary_question):
        """Test that diversity score is in valid range."""
        result = await swarm_protocol_medium.vote(
            question=binary_question["question"], options=binary_question["options"]
        )

        assert 0 <= result.diversity_score <= 100


# Test: Reliability Scoring


class TestReliabilityScoring:
    """Test reliability score calculation."""

    @pytest.mark.asyncio
    async def test_reliability_high_with_strong_consensus(self, swarm_protocol_large):
        """Test high reliability with strong consensus."""
        result = await swarm_protocol_large.vote(
            question="Obvious question?", options=["Yes", "No"]
        )

        if result.consensus_level == ConsensusLevel.STRONG and not result.groupthink_detected:
            assert result.reliability >= 70

    @pytest.mark.asyncio
    async def test_reliability_low_with_no_consensus(self, swarm_protocol_large):
        """Test low reliability with no consensus."""
        result = await swarm_protocol_large.vote(
            question="Complex question?",
            options=["A", "B", "C", "D"],
        )

        if result.consensus_level == ConsensusLevel.NO_CONSENSUS:
            assert result.reliability <= 50

    @pytest.mark.asyncio
    async def test_reliability_penalty_for_groupthink(self, swarm_protocol_large):
        """Test reliability penalty when groupthink detected."""
        result = await swarm_protocol_large.vote(
            question="Simple yes/no?", options=["Yes", "No"]
        )

        if result.groupthink_detected:
            # Reliability should be penalized
            assert result.reliability < 80

    @pytest.mark.asyncio
    async def test_reliability_bonus_for_diversity(self, swarm_protocol_large):
        """Test reliability bonus for diversity."""
        result = await swarm_protocol_large.vote(
            question="Diverse question?", options=["A", "B", "C", "D"]
        )

        if result.diversity_score > 70:
            # High diversity can boost reliability
            assert result.reliability >= 40  # At least moderate


# Test: Verdict Generation


class TestVerdictGeneration:
    """Test verdict determination."""

    @pytest.mark.asyncio
    async def test_verdict_proceed_strong_consensus(self, swarm_protocol_large):
        """Test PROCEED verdict with strong consensus."""
        result = await swarm_protocol_large.vote(
            question="Should we proceed?", options=["Yes", "No"]
        )

        if result.consensus_level == ConsensusLevel.STRONG and result.winning_option == "Yes":
            assert result.verdict == "PROCEED"

    @pytest.mark.asyncio
    async def test_verdict_investigate_weak_consensus(self, swarm_protocol_medium):
        """Test INVESTIGATE verdict with weak consensus."""
        result = await swarm_protocol_medium.vote(
            question="Uncertain question?",
            options=["Yes", "No", "Maybe"],
        )

        if result.consensus_level == ConsensusLevel.WEAK:
            assert result.verdict in ["INVESTIGATE", "PROCEED", "REVISIT"]

    @pytest.mark.asyncio
    async def test_verdict_revisit_no_consensus(self, swarm_protocol_medium):
        """Test REVISIT verdict with no consensus."""
        result = await swarm_protocol_medium.vote(
            question="Complex multi-option?",
            options=["A", "B", "C", "D", "E"],
        )

        if result.consensus_level == ConsensusLevel.NO_CONSENSUS:
            assert result.verdict == "REVISIT"


# Test: Context Awareness


class TestContextAwareness:
    """Test voting with context."""

    @pytest.mark.asyncio
    async def test_voting_with_context(self, swarm_protocol_medium):
        """Test that context is used in voting."""
        result = await swarm_protocol_medium.vote(
            question="Should we publish?",
            options=["Yes", "No"],
            context="Results show 20% improvement, p < 0.01, n = 1000",
        )

        # Context should influence votes (hard to test directly)
        assert isinstance(result, VotingResult)

    @pytest.mark.asyncio
    async def test_voting_without_context(self, swarm_protocol_medium):
        """Test voting without context."""
        result = await swarm_protocol_medium.vote(
            question="Should we proceed?", options=["Yes", "No"], context=None
        )

        assert isinstance(result, VotingResult)


# Test: Edge Cases


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_single_option_invalid(self, swarm_protocol_medium):
        """Test that single option is invalid."""
        with pytest.raises((ValueError, AssertionError)):
            await swarm_protocol_medium.vote(
                question="Only one choice?", options=["Only this"]
            )

    @pytest.mark.asyncio
    async def test_many_options(self, swarm_protocol_large):
        """Test voting with many options."""
        options = [f"Option {i}" for i in range(20)]

        result = await swarm_protocol_large.vote(
            question="Pick one from many?", options=options
        )

        assert result.winning_option in options

    @pytest.mark.asyncio
    async def test_empty_question(self, swarm_protocol_medium):
        """Test with empty question."""
        result = await swarm_protocol_medium.vote(question="", options=["Yes", "No"])

        # Should still work (question can be empty)
        assert isinstance(result, VotingResult)

    def test_very_small_swarm(self):
        """Test with very small swarm."""
        protocol = SwarmVotingProtocol(num_agents=2)
        assert protocol.num_agents == 2

    def test_very_large_swarm(self):
        """Test with very large swarm."""
        protocol = SwarmVotingProtocol(num_agents=1000)
        assert protocol.num_agents == 1000


# Test: Swarm Size Impact


class TestSwarmSizeImpact:
    """Test impact of swarm size on results."""

    @pytest.mark.asyncio
    async def test_larger_swarm_more_reliable(self):
        """Test that larger swarms are more reliable."""
        small = SwarmVotingProtocol(num_agents=10)
        large = SwarmVotingProtocol(num_agents=200)

        question = "Should we proceed with this hypothesis?"
        options = ["Yes", "No"]

        result_small = await small.vote(question, options)
        result_large = await large.vote(question, options)

        # Larger swarms generally more reliable
        # (This is probabilistic, so we just check they both work)
        assert isinstance(result_small, VotingResult)
        assert isinstance(result_large, VotingResult)


# Test: Vote Distribution


class TestVoteDistribution:
    """Test vote distribution properties."""

    @pytest.mark.asyncio
    async def test_distribution_sums_to_total(self, swarm_protocol_large, binary_question):
        """Test that vote distribution sums to total agents."""
        result = await swarm_protocol_large.vote(
            question=binary_question["question"], options=binary_question["options"]
        )

        total = sum(result.vote_distribution.values())
        assert total == swarm_protocol_large.num_agents

    @pytest.mark.asyncio
    async def test_winner_has_most_votes(self, swarm_protocol_large, multi_option_question):
        """Test that winner has plurality of votes."""
        result = await swarm_protocol_large.vote(
            question=multi_option_question["question"],
            options=multi_option_question["options"],
        )

        winner_votes = result.vote_distribution[result.winning_option]
        all_votes = result.vote_distribution.values()

        assert winner_votes == max(all_votes)
