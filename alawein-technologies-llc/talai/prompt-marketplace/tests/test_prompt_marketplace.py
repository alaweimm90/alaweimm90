#!/usr/bin/env python3
"""
Comprehensive test suite for PromptMarketplace

Tests cover:
- Prompt listing and management
- Marketplace search and filtering
- Transaction processing
- Fork and royalty system
- Rating system
- Leaderboard functionality
- User statistics
- Edge cases and error handling
"""

import json
import pytest
import tempfile
from datetime import datetime
from pathlib import Path

from prompt_marketplace.main import (
    PromptMarketplace,
    Prompt,
    Transaction,
    PerformanceReport,
    LeaderboardEntry
)


# Fixtures
@pytest.fixture
def temp_marketplace():
    """Create a temporary marketplace instance"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        # Initialize with empty JSON structure
        json.dump({'prompts': {}, 'transactions': []}, f)
        temp_file = f.name

    marketplace = PromptMarketplace(data_file=temp_file)
    yield marketplace

    # Cleanup
    Path(temp_file).unlink(missing_ok=True)


@pytest.fixture
def marketplace_with_data(temp_marketplace):
    """Create marketplace with sample data"""
    # Add diverse prompts
    prompts = [
        {
            'title': 'Python Code Reviewer',
            'description': 'Reviews Python code for best practices',
            'prompt_text': 'Review this Python code: {code}',
            'category': 'coding',
            'creator_id': 'alice',
            'price': 0.50,
            'tags': ['python', 'review', 'quality']
        },
        {
            'title': 'Creative Story Writer',
            'description': 'Generates creative short stories',
            'prompt_text': 'Write a story about: {topic}',
            'category': 'creative',
            'creator_id': 'bob',
            'price': 0.75,
            'tags': ['story', 'creative', 'fiction']
        },
        {
            'title': 'Data Analyst Helper',
            'description': 'Analyzes datasets and provides insights',
            'prompt_text': 'Analyze this data: {data}',
            'category': 'analysis',
            'creator_id': 'alice',
            'price': 1.00,
            'tags': ['data', 'analytics', 'insights']
        },
        {
            'title': 'Research Paper Summarizer',
            'description': 'Summarizes academic papers',
            'prompt_text': 'Summarize: {paper}',
            'category': 'research',
            'creator_id': 'charlie',
            'price': 0.30,
            'tags': ['research', 'summary', 'academic']
        },
        {
            'title': 'Blog Post Writer',
            'description': 'Creates engaging blog posts',
            'prompt_text': 'Write blog post about: {topic}',
            'category': 'writing',
            'creator_id': 'bob',
            'price': 0.60,
            'tags': ['blog', 'writing', 'content']
        }
    ]

    for p in prompts:
        temp_marketplace.list_prompt(**p)

    return temp_marketplace


# Test Class: Prompt Listing
class TestPromptListing:
    """Tests for listing prompts"""

    def test_list_basic_prompt(self, temp_marketplace):
        """Test listing a basic prompt"""
        prompt = temp_marketplace.list_prompt(
            title='Test Prompt',
            description='A test prompt',
            prompt_text='Do something: {input}',
            category='coding',
            creator_id='user123',
            price=0.50
        )

        assert prompt.prompt_id == 1
        assert prompt.title == 'Test Prompt'
        assert prompt.category == 'coding'
        assert prompt.price_per_use == 0.50
        assert prompt.creator_id == 'user123'
        assert prompt.rating == 0.0
        assert prompt.total_uses == 0
        assert prompt.total_revenue == 0.0
        assert prompt.version == '1.0.0'
        assert prompt.parent_id is None

    def test_list_prompt_with_tags(self, temp_marketplace):
        """Test listing a prompt with tags"""
        prompt = temp_marketplace.list_prompt(
            title='Tagged Prompt',
            description='Prompt with tags',
            prompt_text='Process: {data}',
            category='analysis',
            creator_id='user456',
            price=0.75,
            tags=['ai', 'machine-learning', 'nlp']
        )

        assert len(prompt.tags) == 3
        assert 'ai' in prompt.tags
        assert 'nlp' in prompt.tags

    def test_list_multiple_prompts(self, temp_marketplace):
        """Test listing multiple prompts increments IDs"""
        prompt1 = temp_marketplace.list_prompt(
            title='First', description='First prompt', prompt_text='text1',
            category='coding', creator_id='user1', price=0.25
        )
        prompt2 = temp_marketplace.list_prompt(
            title='Second', description='Second prompt', prompt_text='text2',
            category='writing', creator_id='user2', price=0.50
        )

        assert prompt1.prompt_id == 1
        assert prompt2.prompt_id == 2
        assert len(temp_marketplace.prompts) == 2

    def test_prompt_timestamp_created(self, temp_marketplace):
        """Test that prompts have creation timestamps"""
        prompt = temp_marketplace.list_prompt(
            title='Timestamped', description='desc', prompt_text='text',
            category='creative', creator_id='user', price=0.10
        )

        assert prompt.created_at is not None
        assert prompt.updated_at is not None
        # Parse to ensure valid ISO format
        datetime.fromisoformat(prompt.created_at)
        datetime.fromisoformat(prompt.updated_at)


# Test Class: Search and Filtering
class TestSearch:
    """Tests for marketplace search functionality"""

    def test_search_all_prompts(self, marketplace_with_data):
        """Test searching without filters returns all prompts"""
        results = marketplace_with_data.search()
        assert len(results) == 5

    def test_search_by_category(self, marketplace_with_data):
        """Test filtering by category"""
        coding_results = marketplace_with_data.search(category='coding')
        assert len(coding_results) == 1
        assert coding_results[0].category == 'coding'

        creative_results = marketplace_with_data.search(category='creative')
        assert len(creative_results) == 1
        assert creative_results[0].category == 'creative'

    def test_search_by_tags(self, marketplace_with_data):
        """Test filtering by tags"""
        results = marketplace_with_data.search(tags=['python'])
        assert len(results) == 1
        assert 'python' in results[0].tags

        results = marketplace_with_data.search(tags=['writing', 'creative'])
        assert len(results) >= 1

    def test_search_by_min_rating(self, marketplace_with_data):
        """Test filtering by minimum rating"""
        # Rate some prompts
        marketplace_with_data.rate(1, 4.5, 'user1')
        marketplace_with_data.rate(2, 3.0, 'user2')

        results = marketplace_with_data.search(min_rating=4.0)
        assert len(results) >= 1
        for prompt in results:
            assert prompt.rating >= 4.0

    def test_search_by_max_price(self, marketplace_with_data):
        """Test filtering by maximum price"""
        results = marketplace_with_data.search(max_price=0.50)
        assert len(results) >= 1
        for prompt in results:
            assert prompt.price_per_use <= 0.50

    def test_search_combined_filters(self, marketplace_with_data):
        """Test combining multiple filters"""
        results = marketplace_with_data.search(
            category='coding',
            max_price=1.00,
            min_rating=0.0
        )
        assert len(results) >= 1
        for prompt in results:
            assert prompt.category == 'coding'
            assert prompt.price_per_use <= 1.00

    def test_search_sort_by_rating(self, marketplace_with_data):
        """Test sorting results by rating"""
        # Add ratings
        marketplace_with_data.rate(1, 5.0, 'user1')
        marketplace_with_data.rate(2, 3.0, 'user2')
        marketplace_with_data.rate(3, 4.0, 'user3')

        results = marketplace_with_data.search(sort_by='rating')
        # Should be sorted highest to lowest
        for i in range(len(results) - 1):
            assert results[i].rating >= results[i + 1].rating

    def test_search_sort_by_price(self, marketplace_with_data):
        """Test sorting results by price"""
        results = marketplace_with_data.search(sort_by='price')
        # Should be sorted lowest to highest
        for i in range(len(results) - 1):
            assert results[i].price_per_use <= results[i + 1].price_per_use

    def test_search_sort_by_uses(self, marketplace_with_data):
        """Test sorting results by total uses"""
        # Make some purchases
        marketplace_with_data.buy(1, 'buyer1')
        marketplace_with_data.buy(1, 'buyer2')
        marketplace_with_data.buy(2, 'buyer3')

        results = marketplace_with_data.search(sort_by='uses')
        # Should be sorted highest to lowest
        for i in range(len(results) - 1):
            assert results[i].total_uses >= results[i + 1].total_uses

    def test_search_sort_by_revenue(self, marketplace_with_data):
        """Test sorting results by revenue"""
        # Make purchases
        marketplace_with_data.buy(3, 'buyer1')  # $1.00
        marketplace_with_data.buy(3, 'buyer2')  # $1.00
        marketplace_with_data.buy(2, 'buyer3')  # $0.75

        results = marketplace_with_data.search(sort_by='revenue')
        # Should be sorted highest to lowest
        for i in range(len(results) - 1):
            assert results[i].total_revenue >= results[i + 1].total_revenue


# Test Class: Buying and Transactions
class TestBuying:
    """Tests for buying prompts and transactions"""

    def test_buy_prompt(self, marketplace_with_data):
        """Test basic prompt purchase"""
        transaction = marketplace_with_data.buy(1, 'buyer123')

        assert transaction.prompt_id == 1
        assert transaction.buyer_id == 'buyer123'
        assert transaction.seller_id == 'alice'
        assert transaction.price == 0.50
        assert transaction.royalty_paid == 0.0  # Not a fork
        assert transaction.transaction_id == 1

    def test_buy_updates_prompt_stats(self, marketplace_with_data):
        """Test that buying updates prompt usage and revenue"""
        prompt = marketplace_with_data.prompts[1]
        initial_uses = prompt.total_uses
        initial_revenue = prompt.total_revenue

        marketplace_with_data.buy(1, 'buyer1')

        assert prompt.total_uses == initial_uses + 1
        assert prompt.total_revenue == initial_revenue + 0.50

    def test_buy_multiple_times(self, marketplace_with_data):
        """Test multiple purchases of same prompt"""
        marketplace_with_data.buy(1, 'buyer1')
        marketplace_with_data.buy(1, 'buyer2')
        marketplace_with_data.buy(1, 'buyer3')

        prompt = marketplace_with_data.prompts[1]
        assert prompt.total_uses == 3
        assert prompt.total_revenue == 1.50  # 3 * 0.50

    def test_buy_nonexistent_prompt_raises_error(self, marketplace_with_data):
        """Test buying non-existent prompt raises ValueError"""
        with pytest.raises(ValueError, match="Prompt 999 not found"):
            marketplace_with_data.buy(999, 'buyer123')

    def test_transaction_has_timestamp(self, marketplace_with_data):
        """Test that transactions have timestamps"""
        transaction = marketplace_with_data.buy(1, 'buyer1')

        assert transaction.timestamp is not None
        datetime.fromisoformat(transaction.timestamp)

    def test_multiple_transactions_increment_id(self, marketplace_with_data):
        """Test transaction IDs increment properly"""
        t1 = marketplace_with_data.buy(1, 'buyer1')
        t2 = marketplace_with_data.buy(2, 'buyer2')
        t3 = marketplace_with_data.buy(3, 'buyer3')

        assert t1.transaction_id == 1
        assert t2.transaction_id == 2
        assert t3.transaction_id == 3


# Test Class: Fork System
class TestFork:
    """Tests for forking prompts and royalty system"""

    def test_fork_prompt(self, marketplace_with_data):
        """Test forking an existing prompt"""
        forked = marketplace_with_data.fork(
            prompt_id=1,
            new_creator='forker123',
            modifications='Added error handling'
        )

        assert forked.prompt_id == 6  # Should be new ID
        assert forked.parent_id == 1
        assert forked.creator_id == 'forker123'
        assert 'Modified' in forked.title
        assert forked.version == '1.0.0-fork'
        assert forked.category == marketplace_with_data.prompts[1].category

    def test_fork_preserves_tags(self, marketplace_with_data):
        """Test that forking preserves original tags"""
        original = marketplace_with_data.prompts[1]
        forked = marketplace_with_data.fork(1, 'forker', 'changes')

        assert forked.tags == original.tags

    def test_fork_resets_stats(self, marketplace_with_data):
        """Test that forked prompts have reset statistics"""
        # Build up stats on original
        marketplace_with_data.buy(1, 'buyer1')
        marketplace_with_data.rate(1, 5.0, 'user1')

        forked = marketplace_with_data.fork(1, 'forker', 'improvements')

        assert forked.total_uses == 0
        assert forked.total_revenue == 0.0
        assert forked.rating == 0.0

    def test_fork_royalty_payment(self, marketplace_with_data):
        """Test that buying a fork pays royalty to original creator"""
        forked = marketplace_with_data.fork(1, 'forker', 'mods')
        original = marketplace_with_data.prompts[1]
        original_revenue = original.total_revenue

        transaction = marketplace_with_data.buy(forked.prompt_id, 'buyer1')

        # 20% royalty
        expected_royalty = forked.price_per_use * 0.2
        assert transaction.royalty_paid == expected_royalty
        assert original.total_revenue == original_revenue + expected_royalty

    def test_fork_nonexistent_prompt_raises_error(self, marketplace_with_data):
        """Test forking non-existent prompt raises ValueError"""
        with pytest.raises(ValueError, match="Prompt 999 not found"):
            marketplace_with_data.fork(999, 'forker', 'mods')


# Test Class: Rating System
class TestRating:
    """Tests for prompt rating functionality"""

    def test_rate_prompt(self, marketplace_with_data):
        """Test rating a prompt"""
        marketplace_with_data.rate(1, 4.5, 'user1')
        prompt = marketplace_with_data.prompts[1]
        assert prompt.rating == 4.5

    def test_rate_prompt_average(self, marketplace_with_data):
        """Test rating average calculation"""
        # First rating
        marketplace_with_data.rate(1, 5.0, 'user1')
        # Buy to increment uses for weighted average
        marketplace_with_data.buy(1, 'buyer1')
        # Second rating
        marketplace_with_data.rate(1, 3.0, 'user2')

        prompt = marketplace_with_data.prompts[1]
        # Should be weighted average
        assert 3.0 <= prompt.rating <= 5.0

    def test_rate_nonexistent_prompt_raises_error(self, marketplace_with_data):
        """Test rating non-existent prompt raises ValueError"""
        with pytest.raises(ValueError, match="Prompt 999 not found"):
            marketplace_with_data.rate(999, 4.0, 'user1')

    def test_rate_multiple_prompts(self, marketplace_with_data):
        """Test rating multiple different prompts"""
        marketplace_with_data.rate(1, 5.0, 'user1')
        marketplace_with_data.rate(2, 3.5, 'user2')
        marketplace_with_data.rate(3, 4.0, 'user3')

        assert marketplace_with_data.prompts[1].rating == 5.0
        assert marketplace_with_data.prompts[2].rating == 3.5
        assert marketplace_with_data.prompts[3].rating == 4.0


# Test Class: Leaderboard
class TestLeaderboard:
    """Tests for leaderboard functionality"""

    def test_leaderboard_by_revenue(self, marketplace_with_data):
        """Test leaderboard sorted by revenue"""
        # Generate different revenues
        marketplace_with_data.buy(3, 'buyer1')  # $1.00
        marketplace_with_data.buy(3, 'buyer2')  # $1.00
        marketplace_with_data.buy(2, 'buyer3')  # $0.75

        leaderboard = marketplace_with_data.get_leaderboard(metric='revenue', limit=10)

        assert len(leaderboard) <= 10
        assert leaderboard[0].metric_name == 'Total Revenue'
        # Top should have highest revenue
        for i in range(len(leaderboard) - 1):
            assert leaderboard[i].metric_value >= leaderboard[i + 1].metric_value

    def test_leaderboard_by_uses(self, marketplace_with_data):
        """Test leaderboard sorted by uses"""
        marketplace_with_data.buy(1, 'buyer1')
        marketplace_with_data.buy(1, 'buyer2')
        marketplace_with_data.buy(2, 'buyer3')

        leaderboard = marketplace_with_data.get_leaderboard(metric='uses', limit=10)

        assert leaderboard[0].metric_name == 'Total Uses'
        for i in range(len(leaderboard) - 1):
            assert leaderboard[i].metric_value >= leaderboard[i + 1].metric_value

    def test_leaderboard_by_rating(self, marketplace_with_data):
        """Test leaderboard sorted by rating"""
        marketplace_with_data.rate(1, 5.0, 'user1')
        marketplace_with_data.rate(2, 4.0, 'user2')
        marketplace_with_data.rate(3, 3.5, 'user3')

        leaderboard = marketplace_with_data.get_leaderboard(metric='rating', limit=10)

        assert leaderboard[0].metric_name == 'Average Rating'
        for i in range(len(leaderboard) - 1):
            assert leaderboard[i].metric_value >= leaderboard[i + 1].metric_value

    def test_leaderboard_with_category_filter(self, marketplace_with_data):
        """Test leaderboard filtered by category"""
        leaderboard = marketplace_with_data.get_leaderboard(
            metric='revenue',
            category='coding',
            limit=10
        )

        # Should only include coding prompts
        for entry in leaderboard:
            prompt = marketplace_with_data.prompts[entry.prompt_id]
            assert prompt.category == 'coding'

    def test_leaderboard_limit(self, marketplace_with_data):
        """Test leaderboard respects limit parameter"""
        leaderboard = marketplace_with_data.get_leaderboard(metric='revenue', limit=3)
        assert len(leaderboard) <= 3

    def test_leaderboard_rank_ordering(self, marketplace_with_data):
        """Test leaderboard ranks are sequential"""
        leaderboard = marketplace_with_data.get_leaderboard(metric='uses', limit=5)

        for i, entry in enumerate(leaderboard, 1):
            assert entry.rank == i


# Test Class: Creator Statistics
class TestCreatorStats:
    """Tests for creator statistics"""

    def test_creator_stats_basic(self, marketplace_with_data):
        """Test getting basic creator statistics"""
        stats = marketplace_with_data.get_creator_stats('alice')

        assert stats['total_prompts'] == 2  # alice has 2 prompts
        assert stats['total_revenue'] >= 0.0
        assert stats['total_uses'] >= 0
        assert stats['avg_rating'] >= 0.0

    def test_creator_stats_with_purchases(self, marketplace_with_data):
        """Test creator stats reflect purchases"""
        # Alice owns prompts 1 and 3
        marketplace_with_data.buy(1, 'buyer1')  # $0.50
        marketplace_with_data.buy(3, 'buyer2')  # $1.00

        stats = marketplace_with_data.get_creator_stats('alice')

        assert stats['total_uses'] == 2
        assert stats['total_revenue'] == 1.50

    def test_creator_stats_top_prompt(self, marketplace_with_data):
        """Test creator stats include top prompt"""
        marketplace_with_data.buy(3, 'buyer1')  # Alice's prompt #3

        stats = marketplace_with_data.get_creator_stats('alice')

        assert stats['top_prompt'] is not None
        assert stats['top_prompt']['id'] == 3
        assert 'title' in stats['top_prompt']
        assert 'revenue' in stats['top_prompt']

    def test_creator_stats_nonexistent_creator(self, marketplace_with_data):
        """Test stats for non-existent creator returns zeros"""
        stats = marketplace_with_data.get_creator_stats('nonexistent')

        assert stats['total_prompts'] == 0
        assert stats['total_revenue'] == 0.0
        assert stats['total_uses'] == 0
        assert stats['avg_rating'] == 0.0
        assert stats['top_prompt'] is None

    def test_creator_stats_average_rating(self, marketplace_with_data):
        """Test creator average rating calculation"""
        # Rate alice's prompts
        marketplace_with_data.rate(1, 5.0, 'user1')
        marketplace_with_data.rate(3, 3.0, 'user2')

        stats = marketplace_with_data.get_creator_stats('alice')

        # Average of 5.0 and 3.0
        assert stats['avg_rating'] == 4.0


# Test Class: Persistence
class TestPersistence:
    """Tests for data persistence"""

    def test_save_and_load_prompts(self, temp_marketplace):
        """Test that prompts are saved and loaded correctly"""
        # Add prompt
        temp_marketplace.list_prompt(
            title='Persistent Prompt',
            description='Should persist',
            prompt_text='text',
            category='coding',
            creator_id='user1',
            price=0.50
        )

        # Create new instance with same file
        new_marketplace = PromptMarketplace(data_file=temp_marketplace.data_file)

        assert len(new_marketplace.prompts) == 1
        assert new_marketplace.prompts[1].title == 'Persistent Prompt'

    def test_save_and_load_transactions(self, temp_marketplace):
        """Test that transactions are saved and loaded correctly"""
        temp_marketplace.list_prompt(
            title='Test', description='desc', prompt_text='text',
            category='coding', creator_id='user1', price=0.25
        )
        temp_marketplace.buy(1, 'buyer1')

        # Reload
        new_marketplace = PromptMarketplace(data_file=temp_marketplace.data_file)

        assert len(new_marketplace.transactions) == 1
        assert new_marketplace.transactions[0].buyer_id == 'buyer1'

    def test_persistence_updates_stats(self, temp_marketplace):
        """Test that updated stats persist"""
        temp_marketplace.list_prompt(
            title='Test', description='desc', prompt_text='text',
            category='writing', creator_id='user1', price=0.50
        )
        temp_marketplace.buy(1, 'buyer1')
        temp_marketplace.rate(1, 4.5, 'user1')

        # Reload
        new_marketplace = PromptMarketplace(data_file=temp_marketplace.data_file)
        prompt = new_marketplace.prompts[1]

        assert prompt.total_uses == 1
        assert prompt.total_revenue == 0.50
        assert prompt.rating == 4.5


# Test Class: Trending
class TestTrending:
    """Tests for trending prompts functionality"""

    def test_get_trending(self, marketplace_with_data):
        """Test getting trending prompts"""
        # Make some popular
        marketplace_with_data.buy(1, 'buyer1')
        marketplace_with_data.buy(1, 'buyer2')
        marketplace_with_data.buy(2, 'buyer3')

        trending = marketplace_with_data.get_trending(days=7, limit=10)

        assert len(trending) <= 10
        assert trending[0].total_uses >= trending[-1].total_uses

    def test_trending_limit(self, marketplace_with_data):
        """Test trending respects limit parameter"""
        trending = marketplace_with_data.get_trending(limit=3)
        assert len(trending) <= 3


# Test Class: Edge Cases
class TestEdgeCases:
    """Tests for edge cases and error conditions"""

    def test_empty_marketplace(self, temp_marketplace):
        """Test operations on empty marketplace"""
        results = temp_marketplace.search()
        assert len(results) == 0

        leaderboard = temp_marketplace.get_leaderboard()
        assert len(leaderboard) == 0

    def test_prompt_with_zero_price(self, temp_marketplace):
        """Test creating free prompt"""
        prompt = temp_marketplace.list_prompt(
            title='Free Prompt',
            description='No cost',
            prompt_text='text',
            category='coding',
            creator_id='user1',
            price=0.0
        )

        assert prompt.price_per_use == 0.0

        transaction = temp_marketplace.buy(1, 'buyer1')
        assert transaction.price == 0.0

    def test_search_no_results(self, marketplace_with_data):
        """Test search with no matching results"""
        results = marketplace_with_data.search(
            category='research',
            min_rating=5.0,
            max_price=0.10
        )
        # Might return empty or very few results
        assert isinstance(results, list)

    def test_fork_of_fork(self, marketplace_with_data):
        """Test creating a fork of a forked prompt"""
        # Fork original
        fork1 = marketplace_with_data.fork(1, 'user1', 'first fork')
        # Fork the fork
        fork2 = marketplace_with_data.fork(fork1.prompt_id, 'user2', 'second fork')

        assert fork2.parent_id == fork1.prompt_id
        assert fork1.parent_id == 1

    def test_multiple_buyers_same_prompt(self, marketplace_with_data):
        """Test multiple unique buyers purchasing same prompt"""
        buyers = ['buyer1', 'buyer2', 'buyer3', 'buyer4', 'buyer5']

        for buyer in buyers:
            marketplace_with_data.buy(1, buyer)

        prompt = marketplace_with_data.prompts[1]
        assert prompt.total_uses == 5
        assert len(marketplace_with_data.transactions) == 5

    def test_leaderboard_success_rate_metric(self, marketplace_with_data):
        """Test leaderboard with success_rate metric"""
        # Rate some prompts
        marketplace_with_data.rate(1, 5.0, 'user1')
        marketplace_with_data.rate(2, 4.0, 'user2')

        leaderboard = marketplace_with_data.get_leaderboard(metric='success_rate', limit=5)

        assert len(leaderboard) <= 5
        assert leaderboard[0].metric_name == 'Success Rate'
        # Should use rating as proxy for success rate
        for i in range(len(leaderboard) - 1):
            assert leaderboard[i].metric_value >= leaderboard[i + 1].metric_value

    def test_search_empty_tag_list(self, marketplace_with_data):
        """Test search with empty tag list"""
        results = marketplace_with_data.search(tags=[])
        # Empty tag list should return all prompts
        assert len(results) == 5

    def test_prompt_with_empty_tags(self, temp_marketplace):
        """Test creating prompt with no tags"""
        prompt = temp_marketplace.list_prompt(
            title='No Tags',
            description='Prompt without tags',
            prompt_text='text',
            category='writing',
            creator_id='user1',
            price=0.50
        )

        assert prompt.tags == []

    def test_fork_preserves_price(self, marketplace_with_data):
        """Test that fork preserves original price"""
        original = marketplace_with_data.prompts[1]
        forked = marketplace_with_data.fork(1, 'forker', 'changes')

        assert forked.price_per_use == original.price_per_use

    def test_royalty_calculation_accuracy(self, marketplace_with_data):
        """Test precise royalty calculation"""
        forked = marketplace_with_data.fork(3, 'forker', 'mods')  # $1.00 prompt

        transaction = marketplace_with_data.buy(forked.prompt_id, 'buyer1')

        # Should be exactly 20% = $0.20
        assert transaction.royalty_paid == pytest.approx(0.20, rel=1e-9)

    def test_creator_stats_multiple_creators(self, marketplace_with_data):
        """Test stats for different creators"""
        alice_stats = marketplace_with_data.get_creator_stats('alice')
        bob_stats = marketplace_with_data.get_creator_stats('bob')

        assert alice_stats['total_prompts'] == 2
        assert bob_stats['total_prompts'] == 2

    def test_search_filters_are_cumulative(self, marketplace_with_data):
        """Test that all search filters are applied together"""
        # Add some ratings and purchases
        marketplace_with_data.rate(1, 5.0, 'user1')
        marketplace_with_data.buy(1, 'buyer1')

        results = marketplace_with_data.search(
            category='coding',
            min_rating=4.0,
            max_price=1.00
        )

        # All results should meet ALL criteria
        for prompt in results:
            assert prompt.category == 'coding'
            assert prompt.rating >= 4.0
            assert prompt.price_per_use <= 1.00
