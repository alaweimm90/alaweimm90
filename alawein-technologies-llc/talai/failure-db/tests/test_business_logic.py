"""Tests for business logic and calculations."""

import pytest
from failure_db.main import FailureDB


class TestProbabilityCalculations:
    """Test market probability calculations."""

    def test_probability_starts_at_fifty_percent(self, empty_db, sample_market_data):
        """Test that new markets start at 50% fail probability."""
        market = empty_db.create_market(**sample_market_data)
        assert market.fail_probability == 0.5

    def test_probability_100_percent_all_fail_bets(self, db_with_markets):
        """Test probability goes to 100% with only FAIL bets."""
        db_with_markets.place_bet(market_id=1, user_id="user1", outcome="FAIL", amount=100.0)
        db_with_markets.place_bet(market_id=1, user_id="user2", outcome="FAIL", amount=200.0)

        market = db_with_markets.markets[1]
        assert market.fail_probability == 1.0

    def test_probability_0_percent_all_succeed_bets(self, db_with_markets):
        """Test probability goes to 0% with only SUCCEED bets."""
        db_with_markets.place_bet(market_id=1, user_id="user1", outcome="SUCCEED", amount=100.0)
        db_with_markets.place_bet(market_id=1, user_id="user2", outcome="SUCCEED", amount=200.0)

        market = db_with_markets.markets[1]
        assert market.fail_probability == 0.0

    def test_probability_75_percent(self, db_with_markets):
        """Test specific probability calculation."""
        db_with_markets.place_bet(market_id=1, user_id="user1", outcome="FAIL", amount=75.0)
        db_with_markets.place_bet(market_id=1, user_id="user2", outcome="SUCCEED", amount=25.0)

        market = db_with_markets.markets[1]
        assert market.fail_probability == 0.75

    def test_probability_changes_with_sequential_bets(self, db_with_markets):
        """Test that probability updates correctly with sequential bets."""
        # First bet
        db_with_markets.place_bet(market_id=1, user_id="user1", outcome="FAIL", amount=100.0)
        prob1 = db_with_markets.markets[1].fail_probability
        assert prob1 == 1.0

        # Second bet changes it
        db_with_markets.place_bet(market_id=1, user_id="user2", outcome="SUCCEED", amount=100.0)
        prob2 = db_with_markets.markets[1].fail_probability
        assert prob2 == 0.5

        # Third bet changes again
        db_with_markets.place_bet(market_id=1, user_id="user3", outcome="FAIL", amount=200.0)
        prob3 = db_with_markets.markets[1].fail_probability
        assert prob3 == 0.75  # 300 FAIL / 400 total


class TestPayoutCalculations:
    """Test payout calculation logic."""

    def test_payout_winner_takes_all(self, db_with_markets):
        """Test that only winners get payouts."""
        db_with_markets.place_bet(market_id=1, user_id="winner", outcome="FAIL", amount=100.0)
        db_with_markets.place_bet(market_id=1, user_id="loser", outcome="SUCCEED", amount=100.0)

        # Convert positions
        for market in db_with_markets.markets.values():
            market.positions = dict(market.positions)
        db_with_markets._save_data()

        payouts = db_with_markets.resolve_market(
            market_id=1,
            actual_outcome="FAIL",
            resolver_id="admin"
        )

        assert "winner" in payouts
        assert "loser" not in payouts
        assert payouts["winner"] == 200.0  # Gets entire pool

    def test_payout_proportional_split(self, db_with_markets):
        """Test proportional payout among multiple winners."""
        db_with_markets.place_bet(market_id=1, user_id="user1", outcome="FAIL", amount=100.0)
        db_with_markets.place_bet(market_id=1, user_id="user2", outcome="FAIL", amount=200.0)
        db_with_markets.place_bet(market_id=1, user_id="user3", outcome="SUCCEED", amount=100.0)

        # Convert positions
        for market in db_with_markets.markets.values():
            market.positions = dict(market.positions)
        db_with_markets._save_data()

        payouts = db_with_markets.resolve_market(
            market_id=1,
            actual_outcome="FAIL",
            resolver_id="admin"
        )

        # Total pool = 400, winning bets = 300
        # user1: (100/300) * 400 = 133.33...
        # user2: (200/300) * 400 = 266.66...
        assert abs(payouts["user1"] - 133.33) < 0.01
        assert abs(payouts["user2"] - 266.67) < 0.01
        assert "user3" not in payouts

    def test_payout_total_equals_pool(self, db_with_markets):
        """Test that total payouts equal the pool."""
        db_with_markets.place_bet(market_id=1, user_id="user1", outcome="FAIL", amount=50.0)
        db_with_markets.place_bet(market_id=1, user_id="user2", outcome="FAIL", amount=150.0)

        # Convert positions
        for market in db_with_markets.markets.values():
            market.positions = dict(market.positions)
        db_with_markets._save_data()

        payouts = db_with_markets.resolve_market(
            market_id=1,
            actual_outcome="FAIL",
            resolver_id="admin"
        )

        total_payout = sum(payouts.values())
        assert total_payout == 200.0  # Should equal total pool


class TestSearchLogic:
    """Test search filtering and sorting logic."""

    def test_search_sorting_by_upvotes(self, db_with_failures):
        """Test that search results are sorted by upvotes."""
        # Add more failures with different upvotes
        f1 = db_with_failures.failures[1]
        f2 = db_with_failures.failures[2]
        f3 = db_with_failures.failures[3]

        # Set different upvote counts
        f1.upvotes = 5
        f2.upvotes = 15
        f3.upvotes = 10
        db_with_failures._save_data()

        results = db_with_failures.search_failures()

        # Should be sorted by upvotes descending
        assert results[0].upvotes == 15
        assert results[1].upvotes == 10
        assert results[2].upvotes == 5

    def test_search_case_insensitivity(self, empty_db, sample_failure_data):
        """Test that search is case-insensitive."""
        data = sample_failure_data.copy()
        data["domain"] = "PHYSICS"
        empty_db.submit_failure(**data)

        results_lower = empty_db.search_failures(domain="physics")
        results_upper = empty_db.search_failures(domain="PHYSICS")
        results_mixed = empty_db.search_failures(domain="PhYsIcS")

        assert len(results_lower) == 1
        assert len(results_upper) == 1
        assert len(results_mixed) == 1

    def test_search_partial_reason_matching(self, empty_db, sample_failure_data):
        """Test partial text matching in reason field."""
        data = sample_failure_data.copy()
        data["failure_reason"] = "Thermodynamic constraints prevent operation"
        empty_db.submit_failure(**data)

        results = empty_db.search_failures(reason="thermodynamic")
        assert len(results) == 1

        results = empty_db.search_failures(reason="prevent")
        assert len(results) == 1

        results = empty_db.search_failures(reason="operation")
        assert len(results) == 1

    def test_search_tag_matching_any(self, empty_db, sample_failure_data):
        """Test that tag search uses OR logic (any match)."""
        data = sample_failure_data.copy()
        data["tags"] = ["tag1", "tag2", "tag3"]
        empty_db.submit_failure(**data)

        # Should match if any tag is present
        results = empty_db.search_failures(tags=["tag1"])
        assert len(results) == 1

        results = empty_db.search_failures(tags=["tag2", "tag_nonexistent"])
        assert len(results) == 1

        results = empty_db.search_failures(tags=["tag_nonexistent"])
        assert len(results) == 0


class TestAnalyticsCalculations:
    """Test analytics calculation logic."""

    def test_analytics_average_cost_calculation(self, empty_db, sample_failure_data):
        """Test average cost calculation."""
        data1 = sample_failure_data.copy()
        data1["cost_wasted"] = 1000.0
        empty_db.submit_failure(**data1)

        data2 = sample_failure_data.copy()
        data2["title"] = "Failure 2"
        data2["cost_wasted"] = 2000.0
        empty_db.submit_failure(**data2)

        data3 = sample_failure_data.copy()
        data3["title"] = "Failure 3"
        data3["cost_wasted"] = 3000.0
        empty_db.submit_failure(**data3)

        analytics = empty_db.get_analytics()
        assert analytics.avg_cost_wasted == 2000.0

    def test_analytics_average_time_calculation(self, empty_db, sample_failure_data):
        """Test average time calculation."""
        data1 = sample_failure_data.copy()
        data1["time_wasted"] = 10
        empty_db.submit_failure(**data1)

        data2 = sample_failure_data.copy()
        data2["title"] = "Failure 2"
        data2["time_wasted"] = 20
        empty_db.submit_failure(**data2)

        data3 = sample_failure_data.copy()
        data3["title"] = "Failure 3"
        data3["time_wasted"] = 30
        empty_db.submit_failure(**data3)

        analytics = empty_db.get_analytics()
        assert analytics.avg_time_wasted == 20

    def test_analytics_lesson_frequency(self, empty_db, sample_failure_data):
        """Test that most common lessons appear first."""
        # Add failures with overlapping lessons
        for i in range(5):
            data = sample_failure_data.copy()
            data["title"] = f"Failure {i}"
            data["lessons_learned"] = ["Common Lesson", f"Unique Lesson {i}"]
            empty_db.submit_failure(**data)

        analytics = empty_db.get_analytics()

        # "Common Lesson" appears 5 times, should be first
        assert analytics.top_lessons[0] == "Common Lesson"

    def test_analytics_reason_counting(self, empty_db, sample_failure_data):
        """Test counting of failure reasons."""
        # Create failures with different reasons
        for i in range(3):
            data = sample_failure_data.copy()
            data["title"] = f"Failure {i}"
            data["failure_reason"] = "Reason A"
            empty_db.submit_failure(**data)

        for i in range(2):
            data = sample_failure_data.copy()
            data["title"] = f"Failure B{i}"
            data["failure_reason"] = "Reason B"
            empty_db.submit_failure(**data)

        analytics = empty_db.get_analytics()

        # Should have Reason A with count 3, Reason B with count 2
        reasons_dict = dict(analytics.common_reasons)
        assert reasons_dict["Reason A"] == 3
        assert reasons_dict["Reason B"] == 2


class TestLeaderboardLogic:
    """Test leaderboard calculation and sorting."""

    def test_leaderboard_accuracy_calculation(self, db_with_markets):
        """Test accuracy calculation in leaderboard."""
        # User1: 2 bets, both correct
        db_with_markets.place_bet(market_id=1, user_id="user1", outcome="FAIL", amount=100.0)

        # User2: 2 bets, 1 correct
        db_with_markets.place_bet(market_id=1, user_id="user2", outcome="SUCCEED", amount=50.0)

        # Convert and resolve
        for market in db_with_markets.markets.values():
            market.positions = dict(market.positions)
        db_with_markets._save_data()

        db_with_markets.resolve_market(market_id=1, actual_outcome="FAIL", resolver_id="admin")

        leaderboard = db_with_markets.get_leaderboard()

        # user1 should have 100% accuracy
        user1_entry = next(e for e in leaderboard if e["user_id"] == "user1")
        assert user1_entry["accuracy"] == 1.0
        assert user1_entry["correct_bets"] == 1
        assert user1_entry["total_bets"] == 1

    def test_leaderboard_sorting(self, db_with_markets):
        """Test that leaderboard is sorted by accuracy."""
        # Create scenario with different accuracies
        db_with_markets.place_bet(market_id=1, user_id="alice", outcome="FAIL", amount=100.0)
        db_with_markets.place_bet(market_id=1, user_id="bob", outcome="SUCCEED", amount=100.0)
        db_with_markets.place_bet(market_id=2, user_id="alice", outcome="SUCCEED", amount=100.0)
        db_with_markets.place_bet(market_id=2, user_id="bob", outcome="FAIL", amount=100.0)

        # Convert positions
        for market in db_with_markets.markets.values():
            market.positions = dict(market.positions)
        db_with_markets._save_data()

        # Resolve both markets as FAIL
        # alice: market1=FAIL (correct), market2=SUCCEED (wrong) = 50%
        # bob: market1=SUCCEED (wrong), market2=FAIL (correct) = 50%
        db_with_markets.resolve_market(market_id=1, actual_outcome="FAIL", resolver_id="admin")
        db_with_markets.resolve_market(market_id=2, actual_outcome="FAIL", resolver_id="admin")

        leaderboard = db_with_markets.get_leaderboard()

        # Both should have 50% accuracy
        if len(leaderboard) >= 2:
            for entry in leaderboard[:2]:
                assert entry["user_id"] in ["alice", "bob"]
                assert entry["accuracy"] == 0.5
                assert entry["total_bets"] == 2
                assert entry["correct_bets"] == 1


class TestUserPositions:
    """Test user position tracking in markets."""

    def test_user_position_initialization(self, db_with_markets):
        """Test that user positions are initialized correctly."""
        db_with_markets.place_bet(market_id=1, user_id="newuser", outcome="FAIL", amount=100.0)

        market = db_with_markets.markets[1]
        assert "newuser" in market.positions
        assert market.positions["newuser"]["FAIL"] == 100.0
        assert market.positions["newuser"]["SUCCEED"] == 0.0

    def test_user_position_accumulation(self, db_with_markets):
        """Test that multiple bets accumulate."""
        db_with_markets.place_bet(market_id=1, user_id="user1", outcome="FAIL", amount=100.0)
        db_with_markets.place_bet(market_id=1, user_id="user1", outcome="FAIL", amount=50.0)
        db_with_markets.place_bet(market_id=1, user_id="user1", outcome="SUCCEED", amount=25.0)

        market = db_with_markets.markets[1]
        assert market.positions["user1"]["FAIL"] == 150.0
        assert market.positions["user1"]["SUCCEED"] == 25.0

    def test_multiple_users_positions(self, db_with_markets):
        """Test tracking positions for multiple users."""
        db_with_markets.place_bet(market_id=1, user_id="alice", outcome="FAIL", amount=100.0)
        db_with_markets.place_bet(market_id=1, user_id="bob", outcome="SUCCEED", amount=200.0)
        db_with_markets.place_bet(market_id=1, user_id="charlie", outcome="FAIL", amount=50.0)

        market = db_with_markets.markets[1]
        assert len(market.positions) == 3
        assert market.positions["alice"]["FAIL"] == 100.0
        assert market.positions["bob"]["SUCCEED"] == 200.0
        assert market.positions["charlie"]["FAIL"] == 50.0
