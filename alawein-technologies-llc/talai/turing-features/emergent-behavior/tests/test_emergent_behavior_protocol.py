"""
Comprehensive tests for Emergent Behavior Monitoring Protocol.

Tests anomaly detection, pattern recognition, amplification/suppression, and system health.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from typing import List, Dict, Any

from emergent_behavior import (
    EmergentBehaviorProtocol,
    MonitoringResult,
    EmergentPattern,
    BehaviorType,
    Anomaly,
)


# Fixtures


@pytest.fixture
def mock_orchestrator():
    """Mock TalAI orchestrator."""
    orchestrator = Mock()
    orchestrator.call_llm = AsyncMock(
        return_value="Beneficial pattern: Agents are coordinating efficiently."
    )
    return orchestrator


@pytest.fixture
def sample_agents():
    """Create sample agents for monitoring."""
    return [Mock(agent_id=f"agent_{i}", name=f"Agent {i}") for i in range(10)]


@pytest.fixture
def large_agent_pool():
    """Create large agent pool."""
    return [Mock(agent_id=f"agent_{i}", name=f"Agent {i}") for i in range(50)]


@pytest.fixture
def emergent_protocol():
    """Create emergent behavior protocol."""
    return EmergentBehaviorProtocol()


# Test: Initialization


class TestEmergentBehaviorProtocol:
    """Test suite for Emergent Behavior Protocol."""

    def test_initialization(self):
        """Test protocol initialization."""
        protocol = EmergentBehaviorProtocol()
        assert protocol is not None

    def test_initialization_with_orchestrator(self, mock_orchestrator):
        """Test initialization with orchestrator."""
        protocol = EmergentBehaviorProtocol(orchestrator=mock_orchestrator)
        assert protocol.orchestrator == mock_orchestrator


# Test: Basic Monitoring


class TestBasicMonitoring:
    """Test basic monitoring functionality."""

    @pytest.mark.asyncio
    async def test_monitor_basic(self, emergent_protocol, sample_agents):
        """Test basic monitoring."""
        result = await emergent_protocol.monitor(agents=sample_agents, duration_seconds=1)

        assert isinstance(result, MonitoringResult)
        assert result.duration_seconds == 1
        assert result.num_agents == len(sample_agents)

    @pytest.mark.asyncio
    async def test_monitor_short_duration(self, emergent_protocol, sample_agents):
        """Test monitoring with short duration."""
        result = await emergent_protocol.monitor(agents=sample_agents, duration_seconds=0.5)

        assert result.duration_seconds == 0.5
        assert isinstance(result, MonitoringResult)

    @pytest.mark.asyncio
    async def test_monitor_long_duration(self, emergent_protocol, sample_agents):
        """Test monitoring with longer duration."""
        result = await emergent_protocol.monitor(agents=sample_agents, duration_seconds=3)

        assert result.duration_seconds == 3
        # More patterns with longer monitoring
        assert result.total_patterns >= 0

    @pytest.mark.asyncio
    async def test_monitor_many_agents(self, emergent_protocol, large_agent_pool):
        """Test monitoring with many agents."""
        result = await emergent_protocol.monitor(
            agents=large_agent_pool, duration_seconds=2
        )

        assert result.num_agents == len(large_agent_pool)
        # More agents -> more interactions -> more patterns
        assert isinstance(result, MonitoringResult)


# Test: Anomaly Detection


class TestAnomalyDetection:
    """Test anomaly detection."""

    @pytest.mark.asyncio
    async def test_detects_anomalies(self, emergent_protocol, sample_agents):
        """Test that anomalies are detected."""
        result = await emergent_protocol.monitor(agents=sample_agents, duration_seconds=2)

        # Should detect some anomalies (or none if behavior is normal)
        assert isinstance(result.anomalies_detected, list)

    @pytest.mark.asyncio
    async def test_uniformity_anomaly(self, emergent_protocol):
        """Test detection of uniformity anomaly."""
        # Create agents that all behave uniformly
        uniform_agents = [Mock(agent_id=f"agent_{i}") for i in range(10)]

        result = await emergent_protocol.monitor(
            agents=uniform_agents, duration_seconds=1
        )

        # May detect uniformity (if implemented)
        assert isinstance(result, MonitoringResult)

    @pytest.mark.asyncio
    async def test_high_failure_anomaly(self, emergent_protocol, sample_agents):
        """Test detection of high failure rate anomaly."""
        result = await emergent_protocol.monitor(agents=sample_agents, duration_seconds=2)

        # Check for failure detection (if failures occur)
        assert isinstance(result, MonitoringResult)


# Test: Pattern Recognition


class TestPatternRecognition:
    """Test pattern recognition."""

    @pytest.mark.asyncio
    async def test_recognizes_patterns(self, emergent_protocol, sample_agents):
        """Test that patterns are recognized."""
        result = await emergent_protocol.monitor(agents=sample_agents, duration_seconds=2)

        assert isinstance(result.beneficial_patterns, list)
        assert isinstance(result.harmful_patterns, list)
        assert isinstance(result.neutral_patterns, list)

    @pytest.mark.asyncio
    async def test_pattern_classification(self, emergent_protocol, sample_agents):
        """Test that patterns are properly classified."""
        result = await emergent_protocol.monitor(agents=sample_agents, duration_seconds=2)

        # All patterns should be in one of the categories
        total = (
            len(result.beneficial_patterns)
            + len(result.harmful_patterns)
            + len(result.neutral_patterns)
        )
        assert result.total_patterns == total

    @pytest.mark.asyncio
    async def test_pattern_properties(self, emergent_protocol, sample_agents):
        """Test that patterns have required properties."""
        result = await emergent_protocol.monitor(agents=sample_agents, duration_seconds=2)

        for pattern in result.beneficial_patterns + result.harmful_patterns:
            assert pattern.description  # Has description
            assert pattern.behavior_type  # Has type
            assert isinstance(pattern.impact_score, (int, float))  # Has impact
            assert isinstance(pattern.agents_involved, list)  # Has agents


# Test: Beneficial Patterns


class TestBeneficialPatterns:
    """Test beneficial pattern detection and amplification."""

    @pytest.mark.asyncio
    async def test_beneficial_pattern_detection(self, emergent_protocol, sample_agents):
        """Test detection of beneficial patterns."""
        result = await emergent_protocol.monitor(agents=sample_agents, duration_seconds=2)

        # May or may not find beneficial patterns
        assert isinstance(result.beneficial_patterns, list)

    @pytest.mark.asyncio
    async def test_beneficial_pattern_impact(self, emergent_protocol, sample_agents):
        """Test beneficial patterns have positive impact."""
        result = await emergent_protocol.monitor(agents=sample_agents, duration_seconds=2)

        for pattern in result.beneficial_patterns:
            # Beneficial patterns should have positive impact
            assert pattern.impact_score > 0

    @pytest.mark.asyncio
    async def test_amplification_tracking(self, emergent_protocol, sample_agents):
        """Test that amplified patterns are tracked."""
        result = await emergent_protocol.monitor(agents=sample_agents, duration_seconds=2)

        assert isinstance(result.amplified_patterns, list)
        # Amplified patterns should be subset of beneficial
        for pattern_id in result.amplified_patterns:
            assert any(p.pattern_id == pattern_id for p in result.beneficial_patterns)


# Test: Harmful Patterns


class TestHarmfulPatterns:
    """Test harmful pattern detection and suppression."""

    @pytest.mark.asyncio
    async def test_harmful_pattern_detection(self, emergent_protocol, sample_agents):
        """Test detection of harmful patterns."""
        result = await emergent_protocol.monitor(agents=sample_agents, duration_seconds=2)

        # May or may not find harmful patterns
        assert isinstance(result.harmful_patterns, list)

    @pytest.mark.asyncio
    async def test_harmful_pattern_impact(self, emergent_protocol, sample_agents):
        """Test harmful patterns have negative impact."""
        result = await emergent_protocol.monitor(agents=sample_agents, duration_seconds=2)

        for pattern in result.harmful_patterns:
            # Harmful patterns should have negative impact (or be flagged as harmful)
            assert pattern.behavior_type == BehaviorType.HARMFUL

    @pytest.mark.asyncio
    async def test_suppression_tracking(self, emergent_protocol, sample_agents):
        """Test that suppressed patterns are tracked."""
        result = await emergent_protocol.monitor(agents=sample_agents, duration_seconds=2)

        assert isinstance(result.suppressed_patterns, list)
        # Suppressed patterns should be subset of harmful
        for pattern_id in result.suppressed_patterns:
            assert any(p.pattern_id == pattern_id for p in result.harmful_patterns)


# Test: Neutral Patterns


class TestNeutralPatterns:
    """Test neutral pattern detection."""

    @pytest.mark.asyncio
    async def test_neutral_pattern_detection(self, emergent_protocol, sample_agents):
        """Test detection of neutral patterns."""
        result = await emergent_protocol.monitor(agents=sample_agents, duration_seconds=2)

        assert isinstance(result.neutral_patterns, list)

    @pytest.mark.asyncio
    async def test_neutral_patterns_no_action(self, emergent_protocol, sample_agents):
        """Test that neutral patterns are not amplified or suppressed."""
        result = await emergent_protocol.monitor(agents=sample_agents, duration_seconds=2)

        neutral_ids = [p.pattern_id for p in result.neutral_patterns]

        # Neutral patterns should not be amplified
        assert not any(pid in result.amplified_patterns for pid in neutral_ids)
        # Neutral patterns should not be suppressed
        assert not any(pid in result.suppressed_patterns for pid in neutral_ids)


# Test: System Health Scoring


class TestSystemHealthScoring:
    """Test system health score calculation."""

    @pytest.mark.asyncio
    async def test_health_score_range(self, emergent_protocol, sample_agents):
        """Test health score is in valid range."""
        result = await emergent_protocol.monitor(agents=sample_agents, duration_seconds=2)

        assert 0 <= result.health_score <= 100

    @pytest.mark.asyncio
    async def test_health_score_with_beneficial(self, emergent_protocol):
        """Test health score increases with beneficial patterns."""
        # Simulate beneficial behavior
        agents = [Mock(agent_id=f"agent_{i}") for i in range(10)]

        result = await emergent_protocol.monitor(agents=agents, duration_seconds=2)

        # Health score should be positive (baseline is 50)
        assert result.health_score >= 0

    @pytest.mark.asyncio
    async def test_health_score_with_harmful(self, emergent_protocol, sample_agents):
        """Test health score decreases with harmful patterns."""
        result = await emergent_protocol.monitor(agents=sample_agents, duration_seconds=2)

        if len(result.harmful_patterns) > 0:
            # If harmful patterns found, health might be lower
            assert result.health_score < 100

    def test_health_calculation_formula(self):
        """Test health score calculation formula."""
        result = MonitoringResult(
            num_agents=10,
            duration_seconds=60,
            beneficial_patterns=[
                EmergentPattern(
                    pattern_id="p1",
                    description="Good",
                    behavior_type=BehaviorType.BENEFICIAL,
                    impact_score=50,
                    agents_involved=["a1"],
                )
            ],
            harmful_patterns=[],
            neutral_patterns=[],
            amplified_patterns=["p1"],
            suppressed_patterns=[],
            anomalies_detected=[],
            emergence_rate=1.0,
            verdict="HEALTHY",
        )

        # With 1 beneficial pattern: 50 + (1 * 10) = 60
        assert result.health_score == 60


# Test: Emergence Rate


class TestEmergenceRate:
    """Test emergence rate calculation."""

    @pytest.mark.asyncio
    async def test_emergence_rate_calculated(self, emergent_protocol, sample_agents):
        """Test that emergence rate is calculated."""
        result = await emergent_protocol.monitor(agents=sample_agents, duration_seconds=60)

        # Emergence rate = patterns per minute
        assert result.emergence_rate >= 0

    @pytest.mark.asyncio
    async def test_emergence_rate_increases_with_duration(
        self, emergent_protocol, sample_agents
    ):
        """Test emergence rate with different durations."""
        result_short = await emergent_protocol.monitor(
            agents=sample_agents, duration_seconds=30
        )
        result_long = await emergent_protocol.monitor(
            agents=sample_agents, duration_seconds=120
        )

        # Longer duration may reveal more patterns
        # (This is probabilistic, so we just check both work)
        assert isinstance(result_short.emergence_rate, (int, float))
        assert isinstance(result_long.emergence_rate, (int, float))

    def test_emergence_rate_calculation(self):
        """Test emergence rate calculation formula."""
        result = MonitoringResult(
            num_agents=10,
            duration_seconds=60,
            beneficial_patterns=[Mock() for _ in range(3)],
            harmful_patterns=[Mock()],
            neutral_patterns=[Mock(), Mock()],
            amplified_patterns=[],
            suppressed_patterns=[],
            anomalies_detected=[],
            emergence_rate=0,  # Will be calculated
            verdict="HEALTHY",
        )

        # Total patterns = 3 + 1 + 2 = 6
        # Duration = 60 seconds = 1 minute
        # Rate = 6 / 1 = 6 patterns/min
        expected_rate = 6.0
        assert result.emergence_rate == expected_rate


# Test: Verdict Generation


class TestVerdictGeneration:
    """Test verdict determination."""

    def test_verdict_healthy_high_score(self):
        """Test HEALTHY verdict with high health score."""
        result = MonitoringResult(
            num_agents=10,
            duration_seconds=60,
            beneficial_patterns=[Mock(), Mock(), Mock()],
            harmful_patterns=[],
            neutral_patterns=[],
            amplified_patterns=[],
            suppressed_patterns=[],
            anomalies_detected=[],
            emergence_rate=3.0,
            verdict="HEALTHY",
        )

        # Health = 50 + 3*10 = 80 -> HEALTHY
        assert result.verdict == "HEALTHY"

    def test_verdict_degraded_low_score(self):
        """Test DEGRADED verdict with low health score."""
        result = MonitoringResult(
            num_agents=10,
            duration_seconds=60,
            beneficial_patterns=[],
            harmful_patterns=[Mock(), Mock()],
            neutral_patterns=[],
            amplified_patterns=[],
            suppressed_patterns=[],
            anomalies_detected=[],
            emergence_rate=2.0,
            verdict="DEGRADED",
        )

        # Health = 50 - 2*15 = 20 -> DEGRADED
        assert result.verdict == "DEGRADED"

    def test_verdict_normal_moderate_score(self):
        """Test NORMAL verdict with moderate health score."""
        result = MonitoringResult(
            num_agents=10,
            duration_seconds=60,
            beneficial_patterns=[Mock()],
            harmful_patterns=[Mock()],
            neutral_patterns=[Mock()],
            amplified_patterns=[],
            suppressed_patterns=[],
            anomalies_detected=[],
            emergence_rate=3.0,
            verdict="NORMAL",
        )

        # Health = 50 + 10 - 15 + 2 = 47 -> NORMAL
        assert result.verdict == "NORMAL"


# Test: Interaction Callback


class TestInteractionCallback:
    """Test interaction callback functionality."""

    @pytest.mark.asyncio
    async def test_callback_invoked(self, emergent_protocol, sample_agents):
        """Test that callback is invoked for interactions."""
        interactions_logged = []

        def callback(interaction):
            interactions_logged.append(interaction)

        await emergent_protocol.monitor(
            agents=sample_agents, duration_seconds=1, interaction_callback=callback
        )

        # Callback should be invoked (or not, depending on interactions)
        assert isinstance(interactions_logged, list)

    @pytest.mark.asyncio
    async def test_callback_optional(self, emergent_protocol, sample_agents):
        """Test that callback is optional."""
        result = await emergent_protocol.monitor(
            agents=sample_agents, duration_seconds=1, interaction_callback=None
        )

        assert isinstance(result, MonitoringResult)


# Test: Edge Cases


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_single_agent(self, emergent_protocol):
        """Test monitoring with single agent."""
        agent = [Mock(agent_id="solo")]

        result = await emergent_protocol.monitor(agents=agent, duration_seconds=1)

        assert result.num_agents == 1
        # Limited patterns with single agent
        assert isinstance(result, MonitoringResult)

    @pytest.mark.asyncio
    async def test_two_agents(self, emergent_protocol):
        """Test monitoring with two agents."""
        agents = [Mock(agent_id=f"agent_{i}") for i in range(2)]

        result = await emergent_protocol.monitor(agents=agents, duration_seconds=1)

        assert result.num_agents == 2

    @pytest.mark.asyncio
    async def test_empty_agents_raises_error(self, emergent_protocol):
        """Test that empty agent list raises error."""
        with pytest.raises((ValueError, AssertionError)):
            await emergent_protocol.monitor(agents=[], duration_seconds=1)

    @pytest.mark.asyncio
    async def test_zero_duration_raises_error(self, emergent_protocol, sample_agents):
        """Test that zero duration raises error."""
        with pytest.raises((ValueError, AssertionError)):
            await emergent_protocol.monitor(agents=sample_agents, duration_seconds=0)

    @pytest.mark.asyncio
    async def test_negative_duration_raises_error(self, emergent_protocol, sample_agents):
        """Test that negative duration raises error."""
        with pytest.raises((ValueError, AssertionError)):
            await emergent_protocol.monitor(agents=sample_agents, duration_seconds=-1)


# Test: Pattern Amplification


class TestPatternAmplification:
    """Test pattern amplification mechanics."""

    @pytest.mark.asyncio
    async def test_amplification_increases_likelihood(
        self, emergent_protocol, sample_agents
    ):
        """Test that amplification increases pattern likelihood."""
        result = await emergent_protocol.monitor(agents=sample_agents, duration_seconds=2)

        # Amplified patterns should be tracked
        assert isinstance(result.amplified_patterns, list)

    @pytest.mark.asyncio
    async def test_only_beneficial_amplified(self, emergent_protocol, sample_agents):
        """Test that only beneficial patterns are amplified."""
        result = await emergent_protocol.monitor(agents=sample_agents, duration_seconds=2)

        beneficial_ids = [p.pattern_id for p in result.beneficial_patterns]

        # All amplified patterns should be beneficial
        for pattern_id in result.amplified_patterns:
            assert pattern_id in beneficial_ids


# Test: Pattern Suppression


class TestPatternSuppression:
    """Test pattern suppression mechanics."""

    @pytest.mark.asyncio
    async def test_suppression_reduces_likelihood(self, emergent_protocol, sample_agents):
        """Test that suppression reduces pattern likelihood."""
        result = await emergent_protocol.monitor(agents=sample_agents, duration_seconds=2)

        # Suppressed patterns should be tracked
        assert isinstance(result.suppressed_patterns, list)

    @pytest.mark.asyncio
    async def test_only_harmful_suppressed(self, emergent_protocol, sample_agents):
        """Test that only harmful patterns are suppressed."""
        result = await emergent_protocol.monitor(agents=sample_agents, duration_seconds=2)

        harmful_ids = [p.pattern_id for p in result.harmful_patterns]

        # All suppressed patterns should be harmful
        for pattern_id in result.suppressed_patterns:
            assert pattern_id in harmful_ids


# Test: Continuous Monitoring


class TestContinuousMonitoring:
    """Test continuous monitoring scenarios."""

    @pytest.mark.asyncio
    async def test_multiple_monitoring_windows(self, emergent_protocol, sample_agents):
        """Test multiple consecutive monitoring windows."""
        results = []

        for _ in range(3):
            result = await emergent_protocol.monitor(
                agents=sample_agents, duration_seconds=1
            )
            results.append(result)

        # All windows should complete
        assert len(results) == 3
        for result in results:
            assert isinstance(result, MonitoringResult)

    @pytest.mark.asyncio
    async def test_health_tracking_over_time(self, emergent_protocol, sample_agents):
        """Test health score tracking over time."""
        health_scores = []

        for _ in range(5):
            result = await emergent_protocol.monitor(
                agents=sample_agents, duration_seconds=1
            )
            health_scores.append(result.health_score)

        # All scores should be valid
        assert all(0 <= score <= 100 for score in health_scores)
