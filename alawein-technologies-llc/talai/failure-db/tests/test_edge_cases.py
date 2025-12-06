"""Edge case and integration tests for FailureDB."""

import pytest
import json
from pathlib import Path
from failure_db.main import FailureDB, Failure, PredictionMarket


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_submit_failure_with_zero_cost(self, empty_db, sample_failure_data):
        """Test submitting failure with zero cost."""
        data = sample_failure_data.copy()
        data["cost_wasted"] = 0.0
        failure = empty_db.submit_failure(**data)
        assert failure.cost_wasted == 0.0

    def test_submit_failure_with_zero_time(self, empty_db, sample_failure_data):
        """Test submitting failure with zero time."""
        data = sample_failure_data.copy()
        data["time_wasted"] = 0
        failure = empty_db.submit_failure(**data)
        assert failure.time_wasted == 0

    def test_submit_failure_with_empty_lessons(self, empty_db, sample_failure_data):
        """Test submitting failure with empty lessons list."""
        data = sample_failure_data.copy()
        data["lessons_learned"] = []
        failure = empty_db.submit_failure(**data)
        assert failure.lessons_learned == []

    def test_search_with_all_filters_combined(self, db_with_failures):
        """Test search with all filters applied."""
        results = db_with_failures.search_failures(
            domain="Computer Science",
            reason="limitations",
            tags=["quantum"],
            min_cost=50000.0,
            verified_only=True
        )
        assert len(results) == 1

    def test_search_zero_cost_filter(self, empty_db, sample_failure_data):
        """Test search with min_cost=0."""
        data = sample_failure_data.copy()
        data["cost_wasted"] = 0.0
        empty_db.submit_failure(**data)
        results = empty_db.search_failures(min_cost=0.0)
        assert len(results) == 1

    def test_place_bet_zero_amount(self, db_with_markets):
        """Test placing bet with zero amount."""
        bet = db_with_markets.place_bet(
            market_id=1,
            user_id="user_test",
            outcome="FAIL",
            amount=0.0
        )
        assert bet.amount == 0.0

    def test_place_bet_same_user_both_outcomes(self, db_with_markets):
        """Test user betting on both outcomes."""
        db_with_markets.place_bet(market_id=1, user_id="user1", outcome="FAIL", amount=100.0)
        db_with_markets.place_bet(market_id=1, user_id="user1", outcome="SUCCEED", amount=50.0)

        market = db_with_markets.markets[1]
        assert market.positions["user1"]["FAIL"] == 100.0
        assert market.positions["user1"]["SUCCEED"] == 50.0

    def test_resolve_market_with_no_bets(self, db_with_markets):
        """Test resolving market with no bets."""
        # Create new market without any bets
        market = db_with_markets.create_market(
            idea_title="Test Idea",
            idea_description="Test",
            domain="Test",
            creator_id="test",
            deadline="2030-01-01"
        )
        market.positions = {}
        db_with_markets._save_data()

        payouts = db_with_markets.resolve_market(
            market_id=market.market_id,
            actual_outcome="FAIL",
            resolver_id="admin"
        )
        assert payouts == {}

    def test_resolve_market_succeed_outcome(self, db_with_markets):
        """Test resolving market with SUCCEED outcome."""
        db_with_markets.place_bet(market_id=1, user_id="user1", outcome="SUCCEED", amount=100.0)
        db_with_markets.place_bet(market_id=1, user_id="user2", outcome="FAIL", amount=50.0)

        # Convert positions
        for market in db_with_markets.markets.values():
            market.positions = dict(market.positions)
        db_with_markets._save_data()

        payouts = db_with_markets.resolve_market(
            market_id=1,
            actual_outcome="SUCCEED",
            resolver_id="admin"
        )

        assert "user1" in payouts
        assert "user2" not in payouts
        assert payouts["user1"] == 150.0  # Gets entire pool

    def test_analytics_with_single_failure(self, empty_db, sample_failure_data):
        """Test analytics with only one failure."""
        empty_db.submit_failure(**sample_failure_data)
        analytics = empty_db.get_analytics()

        assert analytics.total_failures == 1
        assert analytics.avg_cost_wasted == sample_failure_data["cost_wasted"]
        assert analytics.avg_time_wasted == sample_failure_data["time_wasted"]

    def test_analytics_duplicate_lessons(self, empty_db, sample_failure_data):
        """Test analytics with duplicate lessons across failures."""
        data1 = sample_failure_data.copy()
        data1["lessons_learned"] = ["Lesson A", "Lesson B"]
        empty_db.submit_failure(**data1)

        data2 = sample_failure_data.copy()
        data2["title"] = "Another Failure"
        data2["lessons_learned"] = ["Lesson A", "Lesson C"]
        empty_db.submit_failure(**data2)

        analytics = empty_db.get_analytics()
        # Lesson A should appear twice
        assert "Lesson A" in analytics.top_lessons

    def test_leaderboard_with_equal_accuracy(self, db_with_markets):
        """Test leaderboard when users have equal accuracy."""
        # Both users bet on FAIL
        db_with_markets.place_bet(market_id=1, user_id="user1", outcome="FAIL", amount=100.0)
        db_with_markets.place_bet(market_id=1, user_id="user2", outcome="FAIL", amount=50.0)

        # Convert positions
        for market in db_with_markets.markets.values():
            market.positions = dict(market.positions)
        db_with_markets._save_data()

        # Resolve as FAIL - both win
        db_with_markets.resolve_market(market_id=1, actual_outcome="FAIL", resolver_id="admin")

        leaderboard = db_with_markets.get_leaderboard()
        # Both should have 100% accuracy
        for entry in leaderboard:
            assert entry["accuracy"] == 1.0

    def test_multiple_bets_on_same_market(self, db_with_markets):
        """Test multiple sequential bets updating probability."""
        initial_prob = db_with_markets.markets[1].fail_probability

        db_with_markets.place_bet(market_id=1, user_id="user1", outcome="FAIL", amount=100.0)
        prob_after_first = db_with_markets.markets[1].fail_probability

        db_with_markets.place_bet(market_id=1, user_id="user2", outcome="FAIL", amount=100.0)
        prob_after_second = db_with_markets.markets[1].fail_probability

        assert initial_prob == 0.5  # Start at 50/50
        assert prob_after_first == 1.0  # 100% after FAIL bet
        assert prob_after_second == 1.0  # Still 100% after another FAIL

    def test_persistence_with_multiple_operations(self, empty_db, sample_failure_data, sample_market_data):
        """Test that multiple operations persist correctly."""
        # Add failure
        failure = empty_db.submit_failure(**sample_failure_data)

        # Create market
        market = empty_db.create_market(**sample_market_data)
        market.positions = {}
        empty_db._save_data()

        # Place bet
        empty_db.place_bet(market_id=market.market_id, user_id="user1", outcome="FAIL", amount=100.0)

        # Reload database
        new_db = FailureDB(data_file=empty_db.data_file)

        assert len(new_db.failures) == 1
        assert len(new_db.markets) == 1
        assert len(new_db.bets) == 1


class TestDataIntegrity:
    """Test data integrity and validation."""

    def test_failure_id_uniqueness(self, empty_db, sample_failure_data):
        """Test that failure IDs are unique."""
        failures = []
        for i in range(10):
            data = sample_failure_data.copy()
            data["title"] = f"Failure {i}"
            failure = empty_db.submit_failure(**data)
            failures.append(failure.failure_id)

        assert len(failures) == len(set(failures))  # All unique

    def test_market_id_uniqueness(self, empty_db, sample_market_data):
        """Test that market IDs are unique."""
        markets = []
        for i in range(10):
            data = sample_market_data.copy()
            data["idea_title"] = f"Idea {i}"
            market = empty_db.create_market(**data)
            market.positions = {}
            markets.append(market.market_id)

        empty_db._save_data()
        assert len(markets) == len(set(markets))  # All unique

    def test_bet_id_uniqueness(self, db_with_markets):
        """Test that bet IDs are unique."""
        bet_ids = []
        for i in range(5):
            bet = db_with_markets.place_bet(
                market_id=1,
                user_id=f"user{i}",
                outcome="FAIL",
                amount=100.0
            )
            bet_ids.append(bet.bet_id)

        assert len(bet_ids) == len(set(bet_ids))  # All unique

    def test_json_structure_integrity(self, empty_db, sample_failure_data):
        """Test that JSON file has correct structure."""
        empty_db.submit_failure(**sample_failure_data)

        with open(empty_db.data_file, 'r') as f:
            data = json.load(f)

        # Check top-level keys
        assert set(data.keys()) == {"failures", "markets", "bets"}

        # Check types
        assert isinstance(data["failures"], dict)
        assert isinstance(data["markets"], dict)
        assert isinstance(data["bets"], list)

        # Check failure structure
        assert "1" in data["failures"]
        failure = data["failures"]["1"]
        required_fields = [
            "failure_id", "title", "domain", "hypothesis", "approach",
            "failure_reason", "evidence", "cost_wasted", "time_wasted",
            "lessons_learned", "replication_attempts", "submitter_id",
            "verified", "upvotes", "tags", "submitted_at"
        ]
        for field in required_fields:
            assert field in failure


class TestComplexScenarios:
    """Test complex multi-step scenarios."""

    def test_full_market_lifecycle(self, empty_db, sample_market_data):
        """Test complete market lifecycle: create -> bet -> resolve."""
        # Create market
        market = empty_db.create_market(**sample_market_data)
        market.positions = {}
        empty_db._save_data()
        assert not market.resolved

        # Multiple users bet
        empty_db.place_bet(market_id=1, user_id="alice", outcome="FAIL", amount=100.0)
        empty_db.place_bet(market_id=1, user_id="bob", outcome="SUCCEED", amount=200.0)
        empty_db.place_bet(market_id=1, user_id="charlie", outcome="FAIL", amount=50.0)

        market = empty_db.markets[1]
        assert market.total_pool == 350.0
        assert market.fail_probability < 0.5  # More SUCCEED bets

        # Convert and save
        for m in empty_db.markets.values():
            m.positions = dict(m.positions)
        empty_db._save_data()

        # Resolve
        payouts = empty_db.resolve_market(market_id=1, actual_outcome="FAIL", resolver_id="admin")

        assert market.resolved
        assert market.actual_outcome == "FAIL"
        assert "alice" in payouts
        assert "charlie" in payouts
        assert "bob" not in payouts

    def test_cross_domain_analytics(self, db_with_failures):
        """Test analytics across different domains."""
        all_analytics = db_with_failures.get_analytics()
        physics_analytics = db_with_failures.get_analytics(domain="Physics")
        ml_analytics = db_with_failures.get_analytics(domain="Machine Learning")

        assert all_analytics.total_failures == 3
        assert physics_analytics.total_failures == 1
        assert ml_analytics.total_failures == 1

    def test_user_betting_history(self, db_with_resolved_markets):
        """Test tracking user's betting history."""
        # Get all bets for a specific user
        user_id = "user123"
        user_bets = [b for b in db_with_resolved_markets.bets if b.user_id == user_id]

        assert len(user_bets) > 0
        for bet in user_bets:
            assert bet.user_id == user_id
            assert bet.amount > 0
            assert bet.outcome in ["FAIL", "SUCCEED"]
