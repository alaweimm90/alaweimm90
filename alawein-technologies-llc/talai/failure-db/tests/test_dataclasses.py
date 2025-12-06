"""Tests for dataclasses and data models."""

import pytest
from datetime import datetime
from failure_db.main import Failure, PredictionMarket, Bet, FailureAnalytics


class TestDataclasses:
    """Test dataclass structures and validation."""

    def test_failure_dataclass_creation(self):
        """Test creating a Failure instance."""
        failure = Failure(
            failure_id=1,
            title="Test Failure",
            domain="Testing",
            hypothesis="Test hypothesis",
            approach="Test approach",
            failure_reason="Test reason",
            evidence="Test evidence",
            cost_wasted=1000.0,
            time_wasted=10,
            lessons_learned=["Lesson 1"],
            replication_attempts=0,
            submitter_id="test_user",
            verified=False,
            upvotes=0,
            tags=["test"],
            submitted_at=datetime.now().isoformat()
        )

        assert failure.failure_id == 1
        assert failure.title == "Test Failure"
        assert failure.cost_wasted == 1000.0
        assert isinstance(failure.lessons_learned, list)
        assert isinstance(failure.tags, list)

    def test_prediction_market_dataclass_creation(self):
        """Test creating a PredictionMarket instance."""
        market = PredictionMarket(
            market_id=1,
            idea_title="Test Idea",
            idea_description="Test Description",
            domain="Testing",
            created_by="test_user",
            deadline="2030-12-31",
            total_pool=0.0,
            fail_probability=0.5,
            positions={},
            resolved=False,
            actual_outcome=None,
            created_at=datetime.now().isoformat()
        )

        assert market.market_id == 1
        assert market.idea_title == "Test Idea"
        assert market.fail_probability == 0.5
        assert market.resolved is False
        assert market.actual_outcome is None
        assert isinstance(market.positions, dict)

    def test_bet_dataclass_creation(self):
        """Test creating a Bet instance."""
        bet = Bet(
            bet_id=1,
            market_id=1,
            user_id="test_user",
            outcome="FAIL",
            amount=100.0,
            probability_at_bet=0.5,
            timestamp=datetime.now().isoformat()
        )

        assert bet.bet_id == 1
        assert bet.market_id == 1
        assert bet.outcome == "FAIL"
        assert bet.amount == 100.0
        assert bet.probability_at_bet == 0.5

    def test_failure_analytics_dataclass_creation(self):
        """Test creating a FailureAnalytics instance."""
        analytics = FailureAnalytics(
            domain="Testing",
            total_failures=10,
            common_reasons=[("Reason 1", 5), ("Reason 2", 3)],
            avg_cost_wasted=5000.0,
            avg_time_wasted=30,
            top_lessons=["Lesson A", "Lesson B"],
            failure_rate_by_approach={"Approach 1": 0.6, "Approach 2": 0.4}
        )

        assert analytics.domain == "Testing"
        assert analytics.total_failures == 10
        assert len(analytics.common_reasons) == 2
        assert analytics.avg_cost_wasted == 5000.0
        assert isinstance(analytics.top_lessons, list)
        assert isinstance(analytics.failure_rate_by_approach, dict)

    def test_failure_with_large_costs(self):
        """Test failure with large cost values."""
        failure = Failure(
            failure_id=1,
            title="Expensive Failure",
            domain="Research",
            hypothesis="Test",
            approach="Test",
            failure_reason="Test",
            evidence="Test",
            cost_wasted=1000000.0,  # $1M
            time_wasted=365,  # 1 year
            lessons_learned=["Expensive lesson"],
            replication_attempts=0,
            submitter_id="user",
            verified=False,
            upvotes=0,
            tags=[],
            submitted_at=datetime.now().isoformat()
        )

        assert failure.cost_wasted == 1000000.0
        assert failure.time_wasted == 365

    def test_market_with_resolved_state(self):
        """Test market in resolved state."""
        market = PredictionMarket(
            market_id=1,
            idea_title="Resolved Idea",
            idea_description="Description",
            domain="Test",
            created_by="user",
            deadline="2025-01-01",
            total_pool=1000.0,
            fail_probability=0.7,
            positions={"user1": {"FAIL": 700.0, "SUCCEED": 0.0}},
            resolved=True,
            actual_outcome="FAIL",
            created_at=datetime.now().isoformat()
        )

        assert market.resolved is True
        assert market.actual_outcome == "FAIL"
        assert market.total_pool == 1000.0

    def test_bet_fail_outcome(self):
        """Test bet with FAIL outcome."""
        bet = Bet(
            bet_id=1,
            market_id=1,
            user_id="user",
            outcome="FAIL",
            amount=50.0,
            probability_at_bet=0.6,
            timestamp=datetime.now().isoformat()
        )

        assert bet.outcome == "FAIL"

    def test_bet_succeed_outcome(self):
        """Test bet with SUCCEED outcome."""
        bet = Bet(
            bet_id=2,
            market_id=1,
            user_id="user",
            outcome="SUCCEED",
            amount=50.0,
            probability_at_bet=0.4,
            timestamp=datetime.now().isoformat()
        )

        assert bet.outcome == "SUCCEED"

    def test_failure_multiple_replication_attempts(self):
        """Test failure with multiple replication attempts."""
        failure = Failure(
            failure_id=1,
            title="Replicated Failure",
            domain="Science",
            hypothesis="Test",
            approach="Test",
            failure_reason="Test",
            evidence="Test",
            cost_wasted=5000.0,
            time_wasted=50,
            lessons_learned=["Reproduced result"],
            replication_attempts=5,
            submitter_id="user",
            verified=True,
            upvotes=10,
            tags=["replicated"],
            submitted_at=datetime.now().isoformat()
        )

        assert failure.replication_attempts == 5
        assert failure.verified is True
        assert failure.upvotes == 10

    def test_analytics_empty_reasons(self):
        """Test analytics with no common reasons."""
        analytics = FailureAnalytics(
            domain="Empty",
            total_failures=0,
            common_reasons=[],
            avg_cost_wasted=0.0,
            avg_time_wasted=0,
            top_lessons=[],
            failure_rate_by_approach={}
        )

        assert analytics.total_failures == 0
        assert len(analytics.common_reasons) == 0
        assert len(analytics.top_lessons) == 0
        assert len(analytics.failure_rate_by_approach) == 0
