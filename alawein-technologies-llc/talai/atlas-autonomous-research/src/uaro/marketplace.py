"""
UARO Primitive Marketplace

"Airbnb for Algorithms" - Publish, discover, and monetize reasoning primitives.

Cycle 31-32: Primitive Marketplace Foundation

Features:
- Publish custom primitives
- Discovery and search
- Usage tracking and analytics
- Rating and review system
- Monetization (pay-per-use, subscription)
- Performance benchmarking

Network effects: More primitives → Better solutions → More users → More primitives
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import hashlib

from uaro.reasoning_primitives import ReasoningPrimitive


class PricingModel(str, Enum):
    """Pricing models for primitives"""

    FREE = "free"
    PAY_PER_USE = "pay_per_use"
    SUBSCRIPTION = "subscription"
    ONE_TIME = "one_time"


@dataclass
class PrimitiveListing:
    """A primitive listed in the marketplace"""

    id: str
    name: str
    description: str
    category: str
    author: str
    primitive: ReasoningPrimitive

    # Pricing
    pricing_model: PricingModel = PricingModel.FREE
    price: float = 0.0  # In credits or USD

    # Metadata
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    compatible_problem_types: List[str] = field(default_factory=list)

    # Performance stats
    total_uses: int = 0
    total_success: int = 0
    avg_solve_time: Optional[float] = None
    avg_rating: Optional[float] = None
    rating_count: int = 0

    # Verification
    verified: bool = False
    test_cases_passed: int = 0
    test_cases_total: int = 0

    # Timestamps
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at

    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_uses == 0:
            return 0.0
        return self.total_success / self.total_uses

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "author": self.author,
            "pricing_model": self.pricing_model,
            "price": self.price,
            "version": self.version,
            "tags": self.tags,
            "compatible_problem_types": self.compatible_problem_types,
            "stats": {
                "total_uses": self.total_uses,
                "success_rate": self.success_rate(),
                "avg_solve_time": self.avg_solve_time,
                "avg_rating": self.avg_rating,
                "rating_count": self.rating_count
            },
            "verified": self.verified,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }


@dataclass
class PrimitiveReview:
    """User review of a primitive"""

    primitive_id: str
    user_id: str
    rating: float  # 1-5 stars
    comment: str
    use_case: str  # What problem they solved
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class UsageRecord:
    """Record of primitive usage"""

    primitive_id: str
    user_id: str
    problem_type: str
    success: bool
    solve_time: float
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class PrimitiveMarketplace:
    """
    Central marketplace for reasoning primitives.

    Enables:
    - Publishing custom primitives
    - Discovering and searching primitives
    - Tracking usage and performance
    - Rating and reviewing
    - Monetization
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize marketplace

        Args:
            db_path: Path to database file (for persistence)
        """
        self.db_path = db_path

        # In-memory storage (would be database in production)
        self.listings: Dict[str, PrimitiveListing] = {}
        self.reviews: Dict[str, List[PrimitiveReview]] = {}  # primitive_id -> reviews
        self.usage_records: List[UsageRecord] = []

        # Load from db if exists
        if db_path:
            self._load_from_db()

    def publish(
        self,
        primitive: ReasoningPrimitive,
        author: str,
        description: str,
        pricing_model: PricingModel = PricingModel.FREE,
        price: float = 0.0,
        tags: Optional[List[str]] = None,
        test_cases: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Publish a primitive to marketplace

        Args:
            primitive: ReasoningPrimitive to publish
            author: Author username/ID
            description: Human-readable description
            pricing_model: How to charge for usage
            price: Price (if not free)
            tags: Tags for discoverability
            test_cases: Test cases to verify primitive works

        Returns:
            Listing ID
        """
        # Generate unique ID
        listing_id = self._generate_id(primitive.name, author)

        # Verify primitive with test cases
        verified = False
        test_pass = 0
        test_total = 0

        if test_cases:
            test_total = len(test_cases)
            test_pass = self._run_test_cases(primitive, test_cases)
            verified = test_pass == test_total

        # Create listing
        listing = PrimitiveListing(
            id=listing_id,
            name=primitive.name,
            description=description,
            category=primitive.category,
            author=author,
            primitive=primitive,
            pricing_model=pricing_model,
            price=price,
            tags=tags or [],
            verified=verified,
            test_cases_passed=test_pass,
            test_cases_total=test_total
        )

        # Store
        self.listings[listing_id] = listing

        # Persist
        if self.db_path:
            self._save_to_db()

        return listing_id

    def discover(
        self,
        problem_type: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        verified_only: bool = False,
        max_price: Optional[float] = None,
        sort_by: str = "rating"  # rating, uses, success_rate, price
    ) -> List[PrimitiveListing]:
        """
        Discover primitives in marketplace

        Args:
            problem_type: Filter by compatible problem type
            category: Filter by category
            tags: Filter by tags
            verified_only: Only show verified primitives
            max_price: Maximum price
            sort_by: How to sort results

        Returns:
            List of matching primitives
        """
        candidates = list(self.listings.values())

        # Apply filters
        if problem_type:
            candidates = [
                p for p in candidates
                if problem_type in p.compatible_problem_types
            ]

        if category:
            candidates = [
                p for p in candidates
                if p.category == category
            ]

        if tags:
            candidates = [
                p for p in candidates
                if any(tag in p.tags for tag in tags)
            ]

        if verified_only:
            candidates = [p for p in candidates if p.verified]

        if max_price is not None:
            candidates = [
                p for p in candidates
                if p.pricing_model == PricingModel.FREE or p.price <= max_price
            ]

        # Sort
        if sort_by == "rating":
            candidates.sort(
                key=lambda p: (p.avg_rating or 0, p.rating_count),
                reverse=True
            )
        elif sort_by == "uses":
            candidates.sort(key=lambda p: p.total_uses, reverse=True)
        elif sort_by == "success_rate":
            candidates.sort(key=lambda p: p.success_rate(), reverse=True)
        elif sort_by == "price":
            candidates.sort(key=lambda p: p.price)

        return candidates

    def get_listing(self, listing_id: str) -> Optional[PrimitiveListing]:
        """Get listing by ID"""
        return self.listings.get(listing_id)

    def get_primitive(self, listing_id: str) -> Optional[ReasoningPrimitive]:
        """Get primitive by listing ID"""
        listing = self.listings.get(listing_id)
        return listing.primitive if listing else None

    def record_usage(
        self,
        listing_id: str,
        user_id: str,
        problem_type: str,
        success: bool,
        solve_time: float
    ):
        """
        Record usage of a primitive

        Args:
            listing_id: Primitive listing ID
            user_id: User who used it
            problem_type: Type of problem solved
            success: Whether it worked
            solve_time: Time taken to solve
        """
        listing = self.listings.get(listing_id)
        if not listing:
            return

        # Update listing stats
        listing.total_uses += 1
        if success:
            listing.total_success += 1

        # Update average solve time
        if listing.avg_solve_time is None:
            listing.avg_solve_time = solve_time
        else:
            # Exponential moving average
            alpha = 0.1
            listing.avg_solve_time = (
                alpha * solve_time +
                (1 - alpha) * listing.avg_solve_time
            )

        listing.updated_at = datetime.now().isoformat()

        # Record usage
        record = UsageRecord(
            primitive_id=listing_id,
            user_id=user_id,
            problem_type=problem_type,
            success=success,
            solve_time=solve_time
        )
        self.usage_records.append(record)

        # Process payment if needed
        if listing.pricing_model == PricingModel.PAY_PER_USE:
            self._process_payment(user_id, listing.author, listing.price)

        # Persist
        if self.db_path:
            self._save_to_db()

    def add_review(
        self,
        listing_id: str,
        user_id: str,
        rating: float,
        comment: str,
        use_case: str
    ):
        """
        Add review for a primitive

        Args:
            listing_id: Primitive listing ID
            user_id: Reviewer user ID
            rating: Rating (1-5 stars)
            comment: Review comment
            use_case: Description of use case
        """
        listing = self.listings.get(listing_id)
        if not listing:
            return

        # Create review
        review = PrimitiveReview(
            primitive_id=listing_id,
            user_id=user_id,
            rating=rating,
            comment=comment,
            use_case=use_case
        )

        # Store review
        if listing_id not in self.reviews:
            self.reviews[listing_id] = []
        self.reviews[listing_id].append(review)

        # Update average rating
        reviews = self.reviews[listing_id]
        listing.avg_rating = sum(r.rating for r in reviews) / len(reviews)
        listing.rating_count = len(reviews)
        listing.updated_at = datetime.now().isoformat()

        # Persist
        if self.db_path:
            self._save_to_db()

    def get_reviews(self, listing_id: str) -> List[PrimitiveReview]:
        """Get all reviews for a primitive"""
        return self.reviews.get(listing_id, [])

    def get_top_primitives(
        self,
        limit: int = 10,
        category: Optional[str] = None
    ) -> List[PrimitiveListing]:
        """
        Get top-rated primitives

        Args:
            limit: Maximum number to return
            category: Optional category filter

        Returns:
            List of top primitives
        """
        return self.discover(
            category=category,
            sort_by="rating",
            verified_only=True
        )[:limit]

    def get_trending_primitives(
        self,
        limit: int = 10,
        days: int = 7
    ) -> List[PrimitiveListing]:
        """
        Get trending primitives (most used recently)

        Args:
            limit: Maximum number to return
            days: Look back this many days

        Returns:
            List of trending primitives
        """
        # Count recent uses per primitive
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        recent_uses = {}
        for record in self.usage_records:
            if record.timestamp >= cutoff:
                recent_uses[record.primitive_id] = recent_uses.get(record.primitive_id, 0) + 1

        # Sort by recent usage
        trending = sorted(
            recent_uses.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]

        # Get listings
        return [
            self.listings[prim_id]
            for prim_id, _ in trending
            if prim_id in self.listings
        ]

    def get_user_revenue(self, user_id: str) -> float:
        """Get total revenue earned by user"""
        # Placeholder - would query payment records in production
        return 0.0

    def get_user_primitives(self, user_id: str) -> List[PrimitiveListing]:
        """Get all primitives published by user"""
        return [
            listing for listing in self.listings.values()
            if listing.author == user_id
        ]

    def search(self, query: str) -> List[PrimitiveListing]:
        """
        Search primitives by name, description, or tags

        Args:
            query: Search query

        Returns:
            List of matching primitives
        """
        query_lower = query.lower()

        matches = []
        for listing in self.listings.values():
            # Search in name
            if query_lower in listing.name.lower():
                matches.append((3.0, listing))  # High weight
                continue

            # Search in description
            if query_lower in listing.description.lower():
                matches.append((2.0, listing))  # Medium weight
                continue

            # Search in tags
            if any(query_lower in tag.lower() for tag in listing.tags):
                matches.append((1.0, listing))  # Low weight

        # Sort by relevance (weight)
        matches.sort(key=lambda x: x[0], reverse=True)

        return [listing for _, listing in matches]

    # ==================== INTERNAL METHODS ====================

    def _generate_id(self, name: str, author: str) -> str:
        """Generate unique listing ID"""
        data = f"{name}:{author}:{datetime.now().isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _run_test_cases(
        self,
        primitive: ReasoningPrimitive,
        test_cases: List[Dict[str, Any]]
    ) -> int:
        """
        Run test cases and count passes

        Args:
            primitive: Primitive to test
            test_cases: List of test cases

        Returns:
            Number of tests passed
        """
        passed = 0

        for test in test_cases:
            try:
                input_data = test.get('input')
                expected_output = test.get('expected_output')

                if not primitive.is_applicable(input_data):
                    continue

                actual_output = primitive.apply(input_data)

                # Simple equality check
                if actual_output == expected_output:
                    passed += 1
            except:
                # Test failed
                pass

        return passed

    def _process_payment(
        self,
        from_user: str,
        to_user: str,
        amount: float,
        platform_fee: float = 0.15
    ):
        """
        Process payment from user to author

        Args:
            from_user: Paying user ID
            to_user: Receiving user ID
            amount: Amount to transfer
            platform_fee: Platform fee percentage
        """
        # Placeholder - would integrate with payment processor
        author_amount = amount * (1 - platform_fee)
        platform_amount = amount * platform_fee

        print(f"Payment: ${amount:.2f} from {from_user} to {to_user}")
        print(f"  Author receives: ${author_amount:.2f}")
        print(f"  Platform fee: ${platform_amount:.2f}")

    def _save_to_db(self):
        """Save marketplace state to database"""
        if not self.db_path:
            return

        # Simple JSON serialization (would use proper DB in production)
        data = {
            "listings": {
                lid: listing.to_dict()
                for lid, listing in self.listings.items()
            },
            "reviews": {
                pid: [
                    {
                        "primitive_id": r.primitive_id,
                        "user_id": r.user_id,
                        "rating": r.rating,
                        "comment": r.comment,
                        "use_case": r.use_case,
                        "created_at": r.created_at
                    }
                    for r in reviews
                ]
                for pid, reviews in self.reviews.items()
            }
        }

        with open(self.db_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_from_db(self):
        """Load marketplace state from database"""
        if not self.db_path:
            return

        try:
            with open(self.db_path, 'r') as f:
                data = json.load(f)

            # Listings would need to reconstruct primitives
            # For now, just load metadata
            self.reviews = {
                pid: [
                    PrimitiveReview(**review_data)
                    for review_data in reviews
                ]
                for pid, reviews in data.get("reviews", {}).items()
            }
        except FileNotFoundError:
            pass


# ==================== CONVENIENCE FUNCTIONS ====================

def create_marketplace(db_path: Optional[str] = None) -> PrimitiveMarketplace:
    """Create a new marketplace instance"""
    return PrimitiveMarketplace(db_path=db_path)
