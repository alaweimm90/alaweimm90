"""
UARO Test Suite - Primitive Marketplace

Tests for marketplace functionality including publish, discover, payments.
Validates network effects and monetization mechanics.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uaro import (
    PrimitiveMarketplace,
    PrimitiveListing,
    PrimitiveReview,
    UsageRecord,
    PricingModel,
    ReasoningPrimitive,
    create_marketplace,
)


class MockPrimitive(ReasoningPrimitive):
    """Mock primitive for testing"""

    def __init__(self, name="mock_primitive"):
        super().__init__(name, "test")

    def apply(self, input_data):
        return input_data

    def is_applicable(self, input_data):
        return True


@pytest.fixture
def marketplace(tmp_path):
    """Create temporary marketplace"""
    db_path = tmp_path / "test_marketplace.json"
    return PrimitiveMarketplace(db_path=str(db_path))


def test_marketplace_initialization(marketplace):
    """Test marketplace initialization"""
    assert marketplace is not None
    assert len(marketplace.listings) == 0
    assert len(marketplace.reviews) == 0


def test_publish_free_primitive(marketplace):
    """Test publishing free primitive"""
    primitive = MockPrimitive("test_free")

    listing_id = marketplace.publish(
        primitive=primitive,
        author="test_user",
        description="Test primitive",
        pricing_model=PricingModel.FREE,
    )

    assert listing_id is not None
    assert listing_id in marketplace.listings

    listing = marketplace.listings[listing_id]
    assert listing.name == "test_free"
    assert listing.author == "test_user"
    assert listing.pricing_model == PricingModel.FREE
    assert listing.price == 0.0


def test_publish_paid_primitive(marketplace):
    """Test publishing paid primitive"""
    primitive = MockPrimitive("test_paid")

    listing_id = marketplace.publish(
        primitive=primitive,
        author="test_user",
        description="Paid primitive",
        pricing_model=PricingModel.PAY_PER_USE,
        price=0.50,
    )

    assert listing_id in marketplace.listings

    listing = marketplace.listings[listing_id]
    assert listing.pricing_model == PricingModel.PAY_PER_USE
    assert listing.price == 0.50


def test_publish_with_tags(marketplace):
    """Test publishing with tags"""
    primitive = MockPrimitive()

    listing_id = marketplace.publish(
        primitive=primitive,
        author="test_user",
        description="Test",
        tags=["optimization", "fast", "experimental"],
    )

    listing = marketplace.listings[listing_id]
    assert len(listing.tags) == 3
    assert "optimization" in listing.tags


def test_discover_by_category(marketplace):
    """Test discovering primitives by category"""
    opt_prim = MockPrimitive("opt_prim")
    opt_prim.category = "optimization"

    search_prim = MockPrimitive("search_prim")
    search_prim.category = "search"

    marketplace.publish(opt_prim, "user1", "Opt primitive")
    marketplace.publish(search_prim, "user2", "Search primitive")

    results = marketplace.discover(category="optimization")

    assert len(results) == 1
    assert results[0].name == "opt_prim"


def test_discover_verified_only(marketplace):
    """Test filtering by verified status"""
    verified_prim = MockPrimitive("verified")
    unverified_prim = MockPrimitive("unverified")

    id1 = marketplace.publish(
        verified_prim,
        "user1",
        "Verified",
        test_cases=[{"input": 1, "expected_output": 1}],
    )
    id2 = marketplace.publish(unverified_prim, "user2", "Unverified")

    marketplace.listings[id1].verified = True

    results = marketplace.discover(verified_only=True)

    assert len(results) == 1
    assert results[0].name == "verified"


def test_discover_by_price(marketplace):
    """Test filtering by maximum price"""
    cheap = MockPrimitive("cheap")
    expensive = MockPrimitive("expensive")

    marketplace.publish(
        cheap, "user1", "Cheap", pricing_model=PricingModel.PAY_PER_USE, price=0.10
    )
    marketplace.publish(
        expensive,
        "user2",
        "Expensive",
        pricing_model=PricingModel.PAY_PER_USE,
        price=5.00,
    )

    results = marketplace.discover(max_price=1.00)

    assert len(results) == 1
    assert results[0].name == "cheap"


def test_discover_sort_by_rating(marketplace):
    """Test sorting by rating"""
    prim1 = MockPrimitive("prim1")
    prim2 = MockPrimitive("prim2")

    id1 = marketplace.publish(prim1, "user1", "Primitive 1")
    id2 = marketplace.publish(prim2, "user2", "Primitive 2")

    marketplace.listings[id1].avg_rating = 4.5
    marketplace.listings[id1].rating_count = 10

    marketplace.listings[id2].avg_rating = 3.0
    marketplace.listings[id2].rating_count = 5

    results = marketplace.discover(sort_by="rating")

    assert results[0].name == "prim1"
    assert results[1].name == "prim2"


def test_get_listing(marketplace):
    """Test getting listing by ID"""
    primitive = MockPrimitive()

    listing_id = marketplace.publish(primitive, "test_user", "Test")

    listing = marketplace.get_listing(listing_id)

    assert listing is not None
    assert listing.id == listing_id


def test_get_primitive(marketplace):
    """Test getting primitive by listing ID"""
    primitive = MockPrimitive("test_prim")

    listing_id = marketplace.publish(primitive, "test_user", "Test")

    retrieved_prim = marketplace.get_primitive(listing_id)

    assert retrieved_prim is not None
    assert retrieved_prim.name == "test_prim"


def test_record_usage(marketplace):
    """Test recording primitive usage"""
    primitive = MockPrimitive()

    listing_id = marketplace.publish(primitive, "test_user", "Test")

    marketplace.record_usage(
        listing_id=listing_id,
        user_id="customer",
        problem_type="test",
        success=True,
        solve_time=1.5,
    )

    listing = marketplace.listings[listing_id]
    assert listing.total_uses == 1
    assert listing.total_success == 1
    assert listing.avg_solve_time == 1.5


def test_usage_tracking_multiple(marketplace):
    """Test multiple usage recordings"""
    primitive = MockPrimitive()

    listing_id = marketplace.publish(primitive, "test_user", "Test")

    marketplace.record_usage(listing_id, "user1", "test", True, 1.0)
    marketplace.record_usage(listing_id, "user2", "test", True, 2.0)
    marketplace.record_usage(listing_id, "user3", "test", False, 3.0)

    listing = marketplace.listings[listing_id]
    assert listing.total_uses == 3
    assert listing.total_success == 2
    assert listing.success_rate() == 2.0 / 3.0


def test_add_review(marketplace):
    """Test adding review to primitive"""
    primitive = MockPrimitive()

    listing_id = marketplace.publish(primitive, "test_user", "Test")

    marketplace.add_review(
        listing_id=listing_id,
        user_id="reviewer",
        rating=4.5,
        comment="Great primitive!",
        use_case="Solved optimization problem",
    )

    listing = marketplace.listings[listing_id]
    assert listing.avg_rating == 4.5
    assert listing.rating_count == 1

    reviews = marketplace.get_reviews(listing_id)
    assert len(reviews) == 1
    assert reviews[0].rating == 4.5


def test_multiple_reviews(marketplace):
    """Test multiple reviews update average"""
    primitive = MockPrimitive()

    listing_id = marketplace.publish(primitive, "test_user", "Test")

    marketplace.add_review(listing_id, "user1", 5.0, "Excellent", "Use case 1")
    marketplace.add_review(listing_id, "user2", 3.0, "Good", "Use case 2")
    marketplace.add_review(listing_id, "user3", 4.0, "Very good", "Use case 3")

    listing = marketplace.listings[listing_id]
    assert listing.rating_count == 3
    assert listing.avg_rating == (5.0 + 3.0 + 4.0) / 3


def test_search(marketplace):
    """Test search functionality"""
    prim1 = MockPrimitive("fast_solver")
    prim2 = MockPrimitive("slow_solver")

    marketplace.publish(prim1, "user1", "Fast optimization solver", tags=["fast"])
    marketplace.publish(prim2, "user2", "Slow but accurate solver")

    results = marketplace.search("fast")

    assert len(results) > 0
    assert any(r.name == "fast_solver" for r in results)


def test_get_top_primitives(marketplace):
    """Test getting top-rated primitives"""
    for i in range(5):
        prim = MockPrimitive(f"prim{i}")
        lid = marketplace.publish(prim, "user", f"Primitive {i}")
        marketplace.listings[lid].avg_rating = 5.0 - i * 0.5
        marketplace.listings[lid].rating_count = 10
        marketplace.listings[lid].verified = True

    top = marketplace.get_top_primitives(limit=3)

    assert len(top) <= 3
    assert top[0].avg_rating >= top[1].avg_rating


def test_get_user_primitives(marketplace):
    """Test getting primitives by author"""
    prim1 = MockPrimitive("prim1")
    prim2 = MockPrimitive("prim2")
    prim3 = MockPrimitive("prim3")

    marketplace.publish(prim1, "alice", "Prim 1")
    marketplace.publish(prim2, "alice", "Prim 2")
    marketplace.publish(prim3, "bob", "Prim 3")

    alice_prims = marketplace.get_user_primitives("alice")

    assert len(alice_prims) == 2
    assert all(p.author == "alice" for p in alice_prims)


def test_create_marketplace_convenience():
    """Test create_marketplace convenience function"""
    marketplace = create_marketplace()

    assert isinstance(marketplace, PrimitiveMarketplace)


def test_listing_success_rate():
    """Test listing success rate calculation"""
    listing = PrimitiveListing(
        id="test",
        name="test",
        description="test",
        category="test",
        author="test",
        primitive=MockPrimitive(),
    )

    assert listing.success_rate() == 0.0

    listing.total_uses = 10
    listing.total_success = 7

    assert listing.success_rate() == 0.7


def test_listing_to_dict():
    """Test listing serialization"""
    listing = PrimitiveListing(
        id="test",
        name="test",
        description="test",
        category="test",
        author="test",
        primitive=MockPrimitive(),
    )

    data = listing.to_dict()

    assert "id" in data
    assert "name" in data
    assert "stats" in data
    assert "verified" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
