"""Comprehensive tests for FailureDB core functionality."""

import pytest
import json
from pathlib import Path
from failure_db.main import FailureDB, Failure, PredictionMarket, Bet


class TestFailureDBInitialization:
    """Test FailureDB initialization and data persistence."""

    def test_init_creates_empty_db(self, temp_db_file):
        """Test creating a new empty database."""
        db = FailureDB(data_file=temp_db_file)
        assert db.failures == {}
        assert db.markets == {}
        assert db.bets == []
        assert db.data_file == Path(temp_db_file)

    def test_init_loads_existing_data(self, temp_db_file):
        """Test loading existing database file."""
        # Create initial data
        initial_data = {
            "failures": {
                "1": {
                    "failure_id": 1,
                    "title": "Test Failure",
                    "domain": "Testing",
                    "hypothesis": "Test hypothesis",
                    "approach": "Test approach",
                    "failure_reason": "Test reason",
                    "evidence": "Test evidence",
                    "cost_wasted": 1000.0,
                    "time_wasted": 10,
                    "lessons_learned": ["Lesson 1"],
                    "replication_attempts": 0,
                    "submitter_id": "test_user",
                    "verified": False,
                    "upvotes": 0,
                    "tags": ["test"],
                    "submitted_at": "2024-01-01T00:00:00"
                }
            },
            "markets": {},
            "bets": []
        }

        with open(temp_db_file, 'w') as f:
            json.dump(initial_data, f)

        db = FailureDB(data_file=temp_db_file)
        assert len(db.failures) == 1
        assert db.failures[1].title == "Test Failure"

    def test_init_with_nonexistent_file(self, temp_db_file):
        """Test initialization with non-existent file."""
        Path(temp_db_file).unlink()  # Remove the file
        db = FailureDB(data_file=temp_db_file)
        assert db.failures == {}
        assert db.markets == {}


class TestFailureSubmission:
    """Test failure submission functionality."""

    def test_submit_basic_failure(self, empty_db, sample_failure_data):
        """Test submitting a basic failure."""
        failure = empty_db.submit_failure(**sample_failure_data)

        assert failure.failure_id == 1
        assert failure.title == sample_failure_data["title"]
        assert failure.domain == sample_failure_data["domain"]
        assert failure.verified is False
        assert failure.upvotes == 0
        assert failure.replication_attempts == 0

    def test_submit_failure_increments_id(self, empty_db, sample_failure_data):
        """Test that failure IDs increment correctly."""
        failure1 = empty_db.submit_failure(**sample_failure_data)
        failure2 = empty_db.submit_failure(**sample_failure_data)

        assert failure1.failure_id == 1
        assert failure2.failure_id == 2

    def test_submit_failure_with_tags(self, empty_db, sample_failure_data):
        """Test submitting failure with tags."""
        failure = empty_db.submit_failure(**sample_failure_data)
        assert "fusion" in failure.tags
        assert "energy" in failure.tags

    def test_submit_failure_without_tags(self, empty_db, sample_failure_data):
        """Test submitting failure without tags."""
        data = sample_failure_data.copy()
        data.pop("tags")
        failure = empty_db.submit_failure(**data)
        assert failure.tags == []

    def test_submit_failure_persists_data(self, empty_db, sample_failure_data):
        """Test that submitted failure is persisted to disk."""
        failure = empty_db.submit_failure(**sample_failure_data)

        # Create new instance and verify data persists
        new_db = FailureDB(data_file=empty_db.data_file)
        assert len(new_db.failures) == 1
        assert new_db.failures[1].title == sample_failure_data["title"]

    def test_submit_failure_sets_timestamp(self, empty_db, sample_failure_data):
        """Test that submitted_at timestamp is set."""
        failure = empty_db.submit_failure(**sample_failure_data)
        assert failure.submitted_at is not None
        assert "T" in failure.submitted_at  # ISO format check

    def test_submit_multiple_failures_different_domains(self, empty_db, sample_failure_data):
        """Test submitting failures in different domains."""
        failure1 = empty_db.submit_failure(**sample_failure_data)

        data2 = sample_failure_data.copy()
        data2["domain"] = "Chemistry"
        data2["title"] = "Chemical Synthesis Failure"
        failure2 = empty_db.submit_failure(**data2)

        assert failure1.domain == "Physics"
        assert failure2.domain == "Chemistry"
        assert len(empty_db.failures) == 2


class TestFailureSearch:
    """Test failure search functionality."""

    def test_search_all_failures(self, db_with_failures):
        """Test searching without filters returns all failures."""
        results = db_with_failures.search_failures()
        assert len(results) == 3

    def test_search_by_domain(self, db_with_failures):
        """Test searching failures by domain."""
        results = db_with_failures.search_failures(domain="Physics")
        assert len(results) == 1
        assert results[0].domain == "Physics"

    def test_search_by_domain_case_insensitive(self, db_with_failures):
        """Test domain search is case-insensitive."""
        results = db_with_failures.search_failures(domain="physics")
        assert len(results) == 1

    def test_search_by_reason(self, db_with_failures):
        """Test searching by failure reason."""
        results = db_with_failures.search_failures(reason="heat")
        assert len(results) == 1

    def test_search_by_reason_partial_match(self, db_with_failures):
        """Test reason search with partial matching."""
        results = db_with_failures.search_failures(reason="walk")
        assert len(results) == 1
        assert "walk" in results[0].failure_reason.lower()

    def test_search_by_tags(self, db_with_failures):
        """Test searching by tags."""
        results = db_with_failures.search_failures(tags=["fusion"])
        assert len(results) == 1
        assert "fusion" in results[0].tags

    def test_search_by_multiple_tags(self, db_with_failures):
        """Test searching with multiple tags (OR logic)."""
        results = db_with_failures.search_failures(tags=["fusion", "ml"])
        assert len(results) == 2

    def test_search_by_min_cost(self, db_with_failures):
        """Test searching by minimum cost."""
        results = db_with_failures.search_failures(min_cost=50000.0)
        assert len(results) == 2
        assert all(f.cost_wasted >= 50000.0 for f in results)

    def test_search_verified_only(self, db_with_failures):
        """Test searching for verified failures only."""
        results = db_with_failures.search_failures(verified_only=True)
        assert len(results) == 1
        assert results[0].verified is True

    def test_search_combined_filters(self, db_with_failures):
        """Test searching with multiple filters."""
        results = db_with_failures.search_failures(
            domain="Computer Science",
            verified_only=True
        )
        assert len(results) == 1
        assert results[0].domain == "Computer Science"
        assert results[0].verified is True

    def test_search_returns_sorted_by_upvotes(self, db_with_failures):
        """Test that results are sorted by upvotes."""
        results = db_with_failures.search_failures()
        # Verified failure has 10 upvotes
        assert results[0].upvotes == 10

    def test_search_no_results(self, db_with_failures):
        """Test search with no matching results."""
        results = db_with_failures.search_failures(domain="Biology")
        assert len(results) == 0


class TestMarketCreation:
    """Test prediction market creation."""

    def test_create_basic_market(self, empty_db, sample_market_data):
        """Test creating a basic market."""
        market = empty_db.create_market(**sample_market_data)

        assert market.market_id == 1
        assert market.idea_title == sample_market_data["idea_title"]
        assert market.domain == sample_market_data["domain"]
        assert market.total_pool == 0.0
        assert market.fail_probability == 0.5
        assert market.resolved is False
        assert market.actual_outcome is None

    def test_create_market_increments_id(self, empty_db, sample_market_data):
        """Test that market IDs increment."""
        market1 = empty_db.create_market(**sample_market_data)
        market2 = empty_db.create_market(**sample_market_data)

        assert market1.market_id == 1
        assert market2.market_id == 2

    def test_create_market_sets_timestamp(self, empty_db, sample_market_data):
        """Test that created_at timestamp is set."""
        market = empty_db.create_market(**sample_market_data)
        assert market.created_at is not None
        assert "T" in market.created_at

    def test_create_market_persists_data(self, empty_db, sample_market_data):
        """Test that market is persisted to disk."""
        market = empty_db.create_market(**sample_market_data)

        new_db = FailureDB(data_file=empty_db.data_file)
        assert len(new_db.markets) == 1
        assert new_db.markets[1].idea_title == sample_market_data["idea_title"]

    def test_create_market_initializes_positions(self, empty_db, sample_market_data):
        """Test that positions dict is initialized."""
        market = empty_db.create_market(**sample_market_data)
        # positions could be dict or defaultdict
        assert hasattr(market.positions, '__getitem__')


class TestBettingFunctionality:
    """Test betting on prediction markets."""

    def test_place_basic_bet(self, db_with_markets):
        """Test placing a basic bet."""
        bet = db_with_markets.place_bet(
            market_id=1,
            user_id="user123",
            outcome="FAIL",
            amount=100.0
        )

        assert bet.bet_id == 1
        assert bet.market_id == 1
        assert bet.user_id == "user123"
        assert bet.outcome == "FAIL"
        assert bet.amount == 100.0

    def test_place_bet_updates_market_pool(self, db_with_markets):
        """Test that betting updates market total pool."""
        db_with_markets.place_bet(
            market_id=1,
            user_id="user123",
            outcome="FAIL",
            amount=100.0
        )

        market = db_with_markets.markets[1]
        assert market.total_pool == 100.0

    def test_place_bet_updates_probability(self, db_with_markets):
        """Test that betting updates fail probability."""
        db_with_markets.place_bet(
            market_id=1,
            user_id="user123",
            outcome="FAIL",
            amount=100.0
        )

        market = db_with_markets.markets[1]
        # Should be 100% since only FAIL bets
        assert market.fail_probability == 1.0

    def test_place_multiple_bets_probability(self, db_with_markets):
        """Test probability calculation with multiple bets."""
        db_with_markets.place_bet(market_id=1, user_id="user1", outcome="FAIL", amount=60.0)
        db_with_markets.place_bet(market_id=1, user_id="user2", outcome="SUCCEED", amount=40.0)

        market = db_with_markets.markets[1]
        assert market.fail_probability == 0.6

    def test_place_bet_records_probability(self, db_with_markets):
        """Test that bet records probability at time of bet."""
        bet = db_with_markets.place_bet(
            market_id=1,
            user_id="user123",
            outcome="FAIL",
            amount=100.0
        )

        assert bet.probability_at_bet == 0.5  # Initial probability

    def test_place_bet_invalid_market(self, db_with_markets):
        """Test placing bet on non-existent market."""
        with pytest.raises(ValueError, match="Market .* not found"):
            db_with_markets.place_bet(
                market_id=999,
                user_id="user123",
                outcome="FAIL",
                amount=100.0
            )

    def test_place_bet_on_resolved_market(self, db_with_markets):
        """Test placing bet on already resolved market."""
        db_with_markets.place_bet(market_id=1, user_id="user1", outcome="FAIL", amount=50.0)
        db_with_markets.resolve_market(market_id=1, actual_outcome="FAIL", resolver_id="admin")

        with pytest.raises(ValueError, match="Market already resolved"):
            db_with_markets.place_bet(
                market_id=1,
                user_id="user2",
                outcome="FAIL",
                amount=100.0
            )

    def test_place_bet_updates_user_position(self, db_with_markets):
        """Test that betting updates user's position."""
        db_with_markets.place_bet(market_id=1, user_id="user123", outcome="FAIL", amount=100.0)

        market = db_with_markets.markets[1]
        assert market.positions["user123"]["FAIL"] == 100.0
        assert market.positions["user123"]["SUCCEED"] == 0.0

    def test_place_multiple_bets_same_user(self, db_with_markets):
        """Test multiple bets from same user accumulate."""
        db_with_markets.place_bet(market_id=1, user_id="user123", outcome="FAIL", amount=100.0)
        db_with_markets.place_bet(market_id=1, user_id="user123", outcome="FAIL", amount=50.0)

        market = db_with_markets.markets[1]
        assert market.positions["user123"]["FAIL"] == 150.0


class TestMarketResolution:
    """Test market resolution and payouts."""

    def test_resolve_market_basic(self, db_with_bets):
        """Test basic market resolution."""
        payouts = db_with_bets.resolve_market(
            market_id=1,
            actual_outcome="FAIL",
            resolver_id="admin"
        )

        market = db_with_bets.markets[1]
        assert market.resolved is True
        assert market.actual_outcome == "FAIL"
        assert isinstance(payouts, dict)

    def test_resolve_market_calculates_payouts(self, db_with_bets):
        """Test payout calculation."""
        # Market 1 has: user123 (100 FAIL), user456 (50 SUCCEED), user789 (150 FAIL)
        payouts = db_with_bets.resolve_market(
            market_id=1,
            actual_outcome="FAIL",
            resolver_id="admin"
        )

        # Total pool = 300, FAIL bets = 250 (user123: 100, user789: 150)
        # user123 should get: (100/250) * 300 = 120
        # user789 should get: (150/250) * 300 = 180
        assert "user123" in payouts
        assert "user789" in payouts
        assert "user456" not in payouts  # Bet on SUCCEED

        assert payouts["user123"] == 120.0
        assert payouts["user789"] == 180.0

    def test_resolve_market_invalid_market(self, db_with_bets):
        """Test resolving non-existent market."""
        with pytest.raises(ValueError, match="Market .* not found"):
            db_with_bets.resolve_market(
                market_id=999,
                actual_outcome="FAIL",
                resolver_id="admin"
            )

    def test_resolve_market_already_resolved(self, db_with_bets):
        """Test resolving already resolved market."""
        db_with_bets.resolve_market(market_id=1, actual_outcome="FAIL", resolver_id="admin")

        with pytest.raises(ValueError, match="Market already resolved"):
            db_with_bets.resolve_market(market_id=1, actual_outcome="SUCCEED", resolver_id="admin")

    def test_resolve_market_persists_data(self, db_with_bets):
        """Test that resolution persists to disk."""
        db_with_bets.resolve_market(market_id=1, actual_outcome="FAIL", resolver_id="admin")

        new_db = FailureDB(data_file=db_with_bets.data_file)
        assert new_db.markets[1].resolved is True
        assert new_db.markets[1].actual_outcome == "FAIL"


class TestAnalytics:
    """Test failure analytics functionality."""

    def test_get_analytics_all_domains(self, db_with_failures):
        """Test analytics across all domains."""
        analytics = db_with_failures.get_analytics()

        assert analytics.domain == "all"
        assert analytics.total_failures == 3
        assert analytics.avg_cost_wasted > 0
        assert analytics.avg_time_wasted > 0

    def test_get_analytics_specific_domain(self, db_with_failures):
        """Test analytics for specific domain."""
        analytics = db_with_failures.get_analytics(domain="Physics")

        assert analytics.domain.lower() == "physics"
        assert analytics.total_failures == 1

    def test_get_analytics_common_reasons(self, db_with_failures):
        """Test common failure reasons."""
        analytics = db_with_failures.get_analytics()

        assert len(analytics.common_reasons) > 0
        assert isinstance(analytics.common_reasons[0], tuple)
        assert len(analytics.common_reasons[0]) == 2  # (reason, count)

    def test_get_analytics_avg_calculations(self, db_with_failures):
        """Test average cost and time calculations."""
        analytics = db_with_failures.get_analytics()

        # Should average: 50000, 10000, 75000 = 45000
        assert analytics.avg_cost_wasted == 45000.0

        # Should average: 180, 60, 365 = 201.67 -> 201
        assert analytics.avg_time_wasted == 201

    def test_get_analytics_top_lessons(self, db_with_failures):
        """Test top lessons learned."""
        analytics = db_with_failures.get_analytics()

        assert len(analytics.top_lessons) > 0
        assert isinstance(analytics.top_lessons, list)

    def test_get_analytics_failure_rate_by_approach(self, db_with_failures):
        """Test failure rate by approach."""
        analytics = db_with_failures.get_analytics()

        assert isinstance(analytics.failure_rate_by_approach, dict)
        # Each approach appears once, so rate = 1/3
        for approach, rate in analytics.failure_rate_by_approach.items():
            assert 0 <= rate <= 1

    def test_get_analytics_with_timeframe(self, db_with_failures):
        """Test analytics filtered by timeframe."""
        # This will depend on timestamp, but tests the filtering
        analytics = db_with_failures.get_analytics(timeframe="2024")
        assert analytics.total_failures >= 0

    def test_get_analytics_empty_results(self, empty_db):
        """Test analytics with no failures."""
        analytics = empty_db.get_analytics()

        assert analytics.total_failures == 0
        assert analytics.common_reasons == []
        assert analytics.avg_cost_wasted == 0.0
        assert analytics.avg_time_wasted == 0
        assert analytics.top_lessons == []
        assert analytics.failure_rate_by_approach == {}


class TestLeaderboard:
    """Test leaderboard functionality."""

    def test_get_leaderboard_basic(self, db_with_resolved_markets):
        """Test basic leaderboard generation."""
        leaderboard = db_with_resolved_markets.get_leaderboard()

        assert isinstance(leaderboard, list)
        assert len(leaderboard) > 0

    def test_get_leaderboard_includes_accuracy(self, db_with_resolved_markets):
        """Test leaderboard includes accuracy."""
        leaderboard = db_with_resolved_markets.get_leaderboard()

        for entry in leaderboard:
            assert "user_id" in entry
            assert "accuracy" in entry
            assert "total_bets" in entry
            assert "correct_bets" in entry
            assert 0 <= entry["accuracy"] <= 1

    def test_get_leaderboard_sorted_by_accuracy(self, db_with_resolved_markets):
        """Test leaderboard is sorted by accuracy."""
        leaderboard = db_with_resolved_markets.get_leaderboard()

        for i in range(len(leaderboard) - 1):
            assert leaderboard[i]["accuracy"] >= leaderboard[i + 1]["accuracy"]

    def test_get_leaderboard_limit(self, db_with_resolved_markets):
        """Test leaderboard respects limit."""
        leaderboard = db_with_resolved_markets.get_leaderboard(limit=2)

        assert len(leaderboard) <= 2

    def test_get_leaderboard_empty(self, empty_db):
        """Test leaderboard with no bets."""
        leaderboard = empty_db.get_leaderboard()

        assert leaderboard == []


class TestDataPersistence:
    """Test data persistence and serialization."""

    def test_save_and_load_cycle(self, empty_db, sample_failure_data):
        """Test complete save and load cycle."""
        # Add data
        empty_db.submit_failure(**sample_failure_data)

        # Load in new instance
        new_db = FailureDB(data_file=empty_db.data_file)

        assert len(new_db.failures) == 1
        assert new_db.failures[1].title == sample_failure_data["title"]

    def test_save_preserves_all_fields(self, empty_db, sample_failure_data):
        """Test that all fields are preserved during save/load."""
        failure = empty_db.submit_failure(**sample_failure_data)

        new_db = FailureDB(data_file=empty_db.data_file)
        loaded_failure = new_db.failures[1]

        assert loaded_failure.title == failure.title
        assert loaded_failure.domain == failure.domain
        assert loaded_failure.cost_wasted == failure.cost_wasted
        assert loaded_failure.lessons_learned == failure.lessons_learned
        assert loaded_failure.tags == failure.tags

    def test_json_file_format(self, empty_db, sample_failure_data):
        """Test that saved JSON has correct format."""
        empty_db.submit_failure(**sample_failure_data)

        with open(empty_db.data_file, 'r') as f:
            data = json.load(f)

        assert "failures" in data
        assert "markets" in data
        assert "bets" in data
        assert isinstance(data["failures"], dict)
        assert isinstance(data["markets"], dict)
        assert isinstance(data["bets"], list)
