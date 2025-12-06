"""Pytest fixtures for FailureDB tests."""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime
from failure_db.main import FailureDB, Failure, PredictionMarket, Bet


@pytest.fixture
def temp_db_file():
    """Create a temporary database file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
        # Write empty JSON structure
        json.dump({"failures": {}, "markets": {}, "bets": []}, f)
    yield temp_path
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def empty_db(temp_db_file):
    """Create an empty FailureDB instance."""
    return FailureDB(data_file=temp_db_file)


@pytest.fixture
def sample_failure_data():
    """Sample failure data for testing."""
    return {
        "title": "Cold Fusion Attempt",
        "domain": "Physics",
        "hypothesis": "Room temperature fusion is possible",
        "approach": "Palladium electrode electrolysis",
        "failure_reason": "No excess heat detected",
        "evidence": "Experimental data shows no anomalies",
        "cost_wasted": 50000.0,
        "time_wasted": 180,
        "lessons_learned": ["Check experimental setup", "Need better controls"],
        "submitter_id": "user123",
        "tags": ["fusion", "energy"]
    }


@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    return {
        "idea_title": "Flying Cars by 2030",
        "idea_description": "Mass-market flying vehicles",
        "domain": "Transportation",
        "creator_id": "user456",
        "deadline": "2030-12-31"
    }


@pytest.fixture
def db_with_failures(empty_db, sample_failure_data):
    """FailureDB with sample failures."""
    # Add multiple failures
    empty_db.submit_failure(**sample_failure_data)

    # Add another failure in different domain
    empty_db.submit_failure(
        title="ML Model Overfitting",
        domain="Machine Learning",
        hypothesis="Deep learning can predict stock prices",
        approach="LSTM neural network",
        failure_reason="Random walk hypothesis",
        evidence="Backtest shows no alpha",
        cost_wasted=10000.0,
        time_wasted=60,
        lessons_learned=["Market efficiency", "Overfitting to noise"],
        submitter_id="user789",
        tags=["ml", "finance"]
    )

    # Add verified failure
    empty_db.submit_failure(
        title="Quantum Computing Algorithm",
        domain="Computer Science",
        hypothesis="Quantum advantage for NP-complete",
        approach="Variational quantum eigensolver",
        failure_reason="Noisy intermediate-scale limitations",
        evidence="Error rates too high",
        cost_wasted=75000.0,
        time_wasted=365,
        lessons_learned=["Need error correction", "Hardware limitations"],
        submitter_id="user101",
        tags=["quantum", "algorithms"]
    )

    # Mark one as verified
    empty_db.failures[3].verified = True
    empty_db.failures[3].upvotes = 10
    empty_db._save_data()

    return empty_db


@pytest.fixture
def db_with_markets(db_with_failures, sample_market_data):
    """FailureDB with sample markets."""
    # Create markets
    market1 = db_with_failures.create_market(**sample_market_data)
    # Convert defaultdict to dict for JSON serialization
    market1.positions = dict(market1.positions)
    db_with_failures._save_data()

    market2 = db_with_failures.create_market(
        idea_title="Brain-Computer Interface for Consumers",
        idea_description="Direct neural link for average users",
        domain="Neuroscience",
        creator_id="user202",
        deadline="2028-06-30"
    )
    market2.positions = dict(market2.positions)
    db_with_failures._save_data()

    return db_with_failures


@pytest.fixture
def db_with_bets(db_with_markets):
    """FailureDB with sample bets."""
    # Place bets on market 1
    db_with_markets.place_bet(market_id=1, user_id="user123", outcome="FAIL", amount=100.0)
    db_with_markets.place_bet(market_id=1, user_id="user456", outcome="SUCCEED", amount=50.0)
    db_with_markets.place_bet(market_id=1, user_id="user789", outcome="FAIL", amount=150.0)

    # Place bets on market 2
    db_with_markets.place_bet(market_id=2, user_id="user123", outcome="FAIL", amount=200.0)
    db_with_markets.place_bet(market_id=2, user_id="user456", outcome="FAIL", amount=100.0)

    # Convert positions to regular dicts for serialization
    for market in db_with_markets.markets.values():
        market.positions = dict(market.positions)
    db_with_markets._save_data()

    return db_with_markets


@pytest.fixture
def db_with_resolved_markets(db_with_bets):
    """FailureDB with resolved markets."""
    # Resolve market 1 as FAIL
    db_with_bets.resolve_market(market_id=1, actual_outcome="FAIL", resolver_id="admin")

    return db_with_bets


@pytest.fixture
def current_timestamp():
    """Get current ISO timestamp."""
    return datetime.now().isoformat()
