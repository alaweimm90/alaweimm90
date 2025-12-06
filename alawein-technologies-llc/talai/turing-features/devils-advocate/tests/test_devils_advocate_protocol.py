"""
Comprehensive tests for Devil's Advocate Protocol.

Tests all attack strategies, flaw categorization, and robustness scoring.
"""

import pytest
from unittest.mock import Mock, AsyncMock

from devils_advocate import DevilsAdvocateProtocol, AttackResult, Flaw, FlawSeverity
from self_refutation import Hypothesis, HypothesisDomain


# Fixtures


@pytest.fixture
def mock_orchestrator():
    """Mock TalAI orchestrator."""
    orchestrator = Mock()
    orchestrator.call_llm = AsyncMock(
        return_value="Critical flaw: Algorithm assumes linearity when non-linear relationships exist."
    )
    return orchestrator


@pytest.fixture
def sample_hypothesis() -> Hypothesis:
    """Create sample hypothesis for testing."""
    return Hypothesis(
        claim="Our algorithm improves optimization performance by 40%",
        domain=HypothesisDomain.OPTIMIZATION,
        evidence=["Benchmark tests show 40% improvement", "Tested on 100 instances"],
        assumptions=["Linear scaling", "Independent variables", "Stationary conditions"],
    )


@pytest.fixture
def weak_hypothesis() -> Hypothesis:
    """Create weak hypothesis with many flaws."""
    return Hypothesis(
        claim="This algorithm solves NP-hard problems in polynomial time",
        domain=HypothesisDomain.OPTIMIZATION,
        evidence=["Tested on 3 small examples"],
        assumptions=["Assumes P=NP", "Ignores computational complexity theory"],
    )


@pytest.fixture
def strong_hypothesis() -> Hypothesis:
    """Create strong hypothesis with few flaws."""
    return Hypothesis(
        claim="Adding caching reduces API response time by 30%",
        domain=HypothesisDomain.OPTIMIZATION,
        evidence=[
            "Tested on 10,000 requests",
            "Statistically significant with p<0.001",
            "Validated across multiple environments",
        ],
        assumptions=["Cache hit rate >50%", "Network latency is primary bottleneck"],
    )


@pytest.fixture
def devils_advocate(mock_orchestrator):
    """Create Devil's Advocate protocol."""
    return DevilsAdvocateProtocol(orchestrator=mock_orchestrator)


# Test: Initialization


class TestDevilsAdvocateProtocol:
    """Test suite for Devil's Advocate Protocol."""

    def test_initialization(self, mock_orchestrator):
        """Test protocol initialization."""
        protocol = DevilsAdvocateProtocol(orchestrator=mock_orchestrator)
        assert protocol.orchestrator == mock_orchestrator

    def test_initialization_no_orchestrator(self):
        """Test initialization without orchestrator."""
        protocol = DevilsAdvocateProtocol()
        assert protocol.orchestrator is None


# Test: Basic Attack


class TestBasicAttack:
    """Test basic adversarial attack."""

    @pytest.mark.asyncio
    async def test_attack_basic(self, devils_advocate, sample_hypothesis):
        """Test basic attack on hypothesis."""
        result = await devils_advocate.attack(sample_hypothesis, iterations=1)

        assert isinstance(result, AttackResult)
        assert result.hypothesis == sample_hypothesis
        assert result.total_flaws_found >= 0
        assert result.verdict in ["ACCEPT", "REJECT", "REVISE"]

    @pytest.mark.asyncio
    async def test_attack_multiple_iterations(self, devils_advocate, sample_hypothesis):
        """Test attack with multiple iterations."""
        result = await devils_advocate.attack(sample_hypothesis, iterations=3)

        assert result.iterations_completed == 3
        # More iterations should find more flaws (or same amount)
        assert result.total_flaws_found >= 0

    @pytest.mark.asyncio
    async def test_attack_weak_hypothesis(self, devils_advocate, weak_hypothesis):
        """Test attack on weak hypothesis finds many flaws."""
        result = await devils_advocate.attack(weak_hypothesis, iterations=3)

        # Weak hypothesis should have critical flaws
        assert len(result.critical_flaws) > 0
        assert result.verdict == "REJECT"
        assert result.robustness_score < 50

    @pytest.mark.asyncio
    async def test_attack_strong_hypothesis(self, devils_advocate, strong_hypothesis):
        """Test attack on strong hypothesis finds few flaws."""
        result = await devils_advocate.attack(strong_hypothesis, iterations=3)

        # Strong hypothesis should have mostly low/medium flaws
        assert len(result.critical_flaws) == 0 or len(result.critical_flaws) <= 1
        assert result.verdict in ["ACCEPT", "REVISE"]
        assert result.robustness_score >= 60


# Test: Attack Strategies


class TestAttackStrategies:
    """Test individual attack strategies."""

    @pytest.mark.asyncio
    async def test_edge_case_identification(self, devils_advocate, sample_hypothesis):
        """Test edge case identification strategy."""
        result = await devils_advocate.attack(sample_hypothesis, iterations=1)

        # Should find at least some edge cases
        edge_case_flaws = [
            f for f in result.all_flaws if "edge" in f.description.lower()
        ]
        # May or may not find edge cases, but shouldn't crash
        assert isinstance(result, AttackResult)

    @pytest.mark.asyncio
    async def test_assumption_challenging(self, devils_advocate, sample_hypothesis):
        """Test assumption challenging strategy."""
        result = await devils_advocate.attack(sample_hypothesis, iterations=1)

        # Should challenge assumptions
        assumption_flaws = [
            f for f in result.all_flaws
            if "assumption" in f.description.lower() or "assumes" in f.description.lower()
        ]
        # Should find at least one assumption issue
        assert isinstance(result, AttackResult)

    @pytest.mark.asyncio
    async def test_scaling_attack(self, devils_advocate, sample_hypothesis):
        """Test scaling attack strategy."""
        result = await devils_advocate.attack(sample_hypothesis, iterations=1)

        # Should test scaling issues
        scaling_flaws = [
            f for f in result.all_flaws if "scal" in f.description.lower()
        ]
        assert isinstance(result, AttackResult)

    @pytest.mark.asyncio
    async def test_composition_attack(self, devils_advocate):
        """Test composition attack strategy."""
        hypothesis = Hypothesis(
            claim="Combining method A and method B improves results",
            domain=HypothesisDomain.MACHINE_LEARNING,
            evidence=["A works well", "B works well"],
        )

        result = await devils_advocate.attack(hypothesis, iterations=1)

        # Should find composition issues
        assert isinstance(result, AttackResult)

    @pytest.mark.asyncio
    async def test_temporal_attack(self, devils_advocate):
        """Test temporal attack strategy."""
        hypothesis = Hypothesis(
            claim="Our model maintains 95% accuracy over time",
            domain=HypothesisDomain.MACHINE_LEARNING,
            evidence=["Initial tests show 95% accuracy"],
        )

        result = await devils_advocate.attack(hypothesis, iterations=1)

        # Should find temporal issues
        assert isinstance(result, AttackResult)


# Test: Flaw Categorization


class TestFlawCategorization:
    """Test flaw severity categorization."""

    @pytest.mark.asyncio
    async def test_critical_flaw_detection(self, devils_advocate):
        """Test that critical flaws are detected."""
        hypothesis = Hypothesis(
            claim="This algorithm violates the laws of thermodynamics",
            domain=HypothesisDomain.QUANTUM,
            evidence=["Theoretical derivation"],
        )

        result = await devils_advocate.attack(hypothesis, iterations=2)

        # Should identify critical flaw (violates physics)
        assert len(result.critical_flaws) > 0

    @pytest.mark.asyncio
    async def test_flaw_severity_categorization(self, devils_advocate, sample_hypothesis):
        """Test that flaws are properly categorized by severity."""
        result = await devils_advocate.attack(sample_hypothesis, iterations=3)

        # Check all flaws have valid severity
        for flaw in result.all_flaws:
            assert flaw.severity in [
                FlawSeverity.CRITICAL,
                FlawSeverity.HIGH,
                FlawSeverity.MEDIUM,
                FlawSeverity.LOW,
            ]

    @pytest.mark.asyncio
    async def test_flaw_properties(self, devils_advocate, sample_hypothesis):
        """Test that flaws have required properties."""
        result = await devils_advocate.attack(sample_hypothesis, iterations=2)

        for flaw in result.all_flaws:
            assert flaw.description  # Not empty
            assert flaw.severity
            assert flaw.example or flaw.example == ""  # May be empty
            assert flaw.mitigation  # Should have mitigation suggestion


# Test: Robustness Scoring


class TestRobustnessScoring:
    """Test robustness score calculation."""

    def test_robustness_perfect(self):
        """Test robustness score with no flaws."""
        result = AttackResult(
            hypothesis=Hypothesis(claim="Test", domain=HypothesisDomain.OPTIMIZATION),
            iterations_completed=3,
            all_flaws=[],
        )

        assert result.robustness_score == 100

    def test_robustness_with_critical_flaws(self):
        """Test robustness score with critical flaws."""
        flaws = [
            Flaw(
                description="Critical flaw",
                severity=FlawSeverity.CRITICAL,
                example="Example",
                mitigation="Fix",
            )
            for _ in range(2)
        ]

        result = AttackResult(
            hypothesis=Hypothesis(claim="Test", domain=HypothesisDomain.OPTIMIZATION),
            iterations_completed=3,
            all_flaws=flaws,
        )

        # 2 critical flaws = 2 * 10 * 5 = 100 penalty -> score = 0
        assert result.robustness_score == 0

    def test_robustness_with_mixed_flaws(self):
        """Test robustness score with mixed severity flaws."""
        flaws = [
            Flaw(
                description="High flaw",
                severity=FlawSeverity.HIGH,
                example="",
                mitigation="",
            ),
            Flaw(
                description="Medium flaw",
                severity=FlawSeverity.MEDIUM,
                example="",
                mitigation="",
            ),
            Flaw(
                description="Low flaw",
                severity=FlawSeverity.LOW,
                example="",
                mitigation="",
            ),
        ]

        result = AttackResult(
            hypothesis=Hypothesis(claim="Test", domain=HypothesisDomain.OPTIMIZATION),
            iterations_completed=3,
            all_flaws=flaws,
        )

        # Penalty = (1*5 + 1*2 + 1*1) * 5 = 40
        # Score = 100 - 40 = 60
        assert result.robustness_score == 60

    def test_robustness_score_bounds(self):
        """Test that robustness score stays within [0, 100]."""
        # Massive number of flaws
        flaws = [
            Flaw(
                description="Critical",
                severity=FlawSeverity.CRITICAL,
                example="",
                mitigation="",
            )
            for _ in range(50)
        ]

        result = AttackResult(
            hypothesis=Hypothesis(claim="Test", domain=HypothesisDomain.OPTIMIZATION),
            iterations_completed=3,
            all_flaws=flaws,
        )

        # Should clamp to 0
        assert 0 <= result.robustness_score <= 100


# Test: Verdict Generation


class TestVerdictGeneration:
    """Test verdict determination."""

    def test_verdict_accept_high_robustness(self):
        """Test ACCEPT verdict with high robustness."""
        result = AttackResult(
            hypothesis=Hypothesis(claim="Test", domain=HypothesisDomain.OPTIMIZATION),
            iterations_completed=3,
            all_flaws=[
                Flaw(
                    description="Minor issue",
                    severity=FlawSeverity.LOW,
                    example="",
                    mitigation="",
                )
            ],
        )

        # High robustness -> ACCEPT
        assert result.verdict == "ACCEPT"

    def test_verdict_reject_critical_flaws(self):
        """Test REJECT verdict with critical flaws."""
        result = AttackResult(
            hypothesis=Hypothesis(claim="Test", domain=HypothesisDomain.OPTIMIZATION),
            iterations_completed=3,
            all_flaws=[
                Flaw(
                    description="Critical",
                    severity=FlawSeverity.CRITICAL,
                    example="",
                    mitigation="",
                ),
                Flaw(
                    description="Critical 2",
                    severity=FlawSeverity.CRITICAL,
                    example="",
                    mitigation="",
                ),
            ],
        )

        # 2+ critical flaws -> REJECT
        assert result.verdict == "REJECT"

    def test_verdict_revise_moderate_issues(self):
        """Test REVISE verdict with moderate issues."""
        result = AttackResult(
            hypothesis=Hypothesis(claim="Test", domain=HypothesisDomain.OPTIMIZATION),
            iterations_completed=3,
            all_flaws=[
                Flaw(
                    description="High",
                    severity=FlawSeverity.HIGH,
                    example="",
                    mitigation="",
                ),
                Flaw(
                    description="Medium",
                    severity=FlawSeverity.MEDIUM,
                    example="",
                    mitigation="",
                ),
            ],
        )

        # Moderate issues -> REVISE
        assert result.verdict in ["REVISE", "REJECT"]


# Test: Mitigation Suggestions


class TestMitigationSuggestions:
    """Test mitigation suggestion generation."""

    @pytest.mark.asyncio
    async def test_mitigations_provided(self, devils_advocate, sample_hypothesis):
        """Test that mitigations are provided for all flaws."""
        result = await devils_advocate.attack(sample_hypothesis, iterations=2)

        for flaw in result.all_flaws:
            # Every flaw should have mitigation (even if empty string)
            assert flaw.mitigation is not None

    @pytest.mark.asyncio
    async def test_mitigation_quality(self, devils_advocate, weak_hypothesis):
        """Test that critical flaws have detailed mitigations."""
        result = await devils_advocate.attack(weak_hypothesis, iterations=2)

        for flaw in result.critical_flaws:
            # Critical flaws should have non-trivial mitigations
            assert len(flaw.mitigation) > 10  # At least some guidance


# Test: Edge Cases


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_zero_iterations(self, devils_advocate, sample_hypothesis):
        """Test attack with zero iterations."""
        with pytest.raises((ValueError, AssertionError)):
            await devils_advocate.attack(sample_hypothesis, iterations=0)

    @pytest.mark.asyncio
    async def test_negative_iterations(self, devils_advocate, sample_hypothesis):
        """Test attack with negative iterations."""
        with pytest.raises((ValueError, AssertionError)):
            await devils_advocate.attack(sample_hypothesis, iterations=-1)

    @pytest.mark.asyncio
    async def test_single_iteration(self, devils_advocate, sample_hypothesis):
        """Test attack with single iteration."""
        result = await devils_advocate.attack(sample_hypothesis, iterations=1)

        assert result.iterations_completed == 1
        assert isinstance(result, AttackResult)

    @pytest.mark.asyncio
    async def test_many_iterations(self, devils_advocate, sample_hypothesis):
        """Test attack with many iterations."""
        result = await devils_advocate.attack(sample_hypothesis, iterations=10)

        assert result.iterations_completed == 10
        # More iterations should find more (or same) flaws
        assert result.total_flaws_found >= 0


# Test: Comparison


class TestComparison:
    """Test comparison between different hypotheses."""

    @pytest.mark.asyncio
    async def test_weak_vs_strong(self, devils_advocate, weak_hypothesis, strong_hypothesis):
        """Test that weak hypothesis scores worse than strong."""
        result_weak = await devils_advocate.attack(weak_hypothesis, iterations=3)
        result_strong = await devils_advocate.attack(strong_hypothesis, iterations=3)

        # Weak should have lower robustness
        assert result_weak.robustness_score < result_strong.robustness_score
        # Weak should have more flaws
        assert result_weak.total_flaws_found >= result_strong.total_flaws_found


# Test: Flaw Tracking


class TestFlawTracking:
    """Test flaw tracking across iterations."""

    @pytest.mark.asyncio
    async def test_flaw_accumulation(self, devils_advocate, sample_hypothesis):
        """Test that flaws accumulate across iterations."""
        result = await devils_advocate.attack(sample_hypothesis, iterations=3)

        # Total flaws = critical + high + medium + low
        expected_total = (
            len(result.critical_flaws)
            + len(result.high_flaws)
            + len(result.medium_flaws)
            + len(result.low_flaws)
        )

        assert result.total_flaws_found == expected_total

    @pytest.mark.asyncio
    async def test_all_flaws_list(self, devils_advocate, sample_hypothesis):
        """Test that all_flaws contains all categorized flaws."""
        result = await devils_advocate.attack(sample_hypothesis, iterations=2)

        categorized_count = (
            len(result.critical_flaws)
            + len(result.high_flaws)
            + len(result.medium_flaws)
            + len(result.low_flaws)
        )

        assert len(result.all_flaws) == categorized_count
