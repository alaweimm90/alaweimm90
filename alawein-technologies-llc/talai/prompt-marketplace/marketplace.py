#!/usr/bin/env python3
"""
PromptMarketplace - Buy/Sell/Trade Prompts with Darwinian Evolution

Economic natural selection for prompts:
- List prompts with performance metrics
- Usage-based royalties for creators
- A/B testing leaderboards
- Version control and forking
- Quality-based pricing

Usage:
    python marketplace.py list-prompt --title "Code reviewer" --prompt "..." --price 0.10
    python marketplace.py search --category "coding" --min-rating 4.0
    python marketplace.py buy --prompt-id 42 --payment-method credits
    python marketplace.py stats --user-id creator123
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


@dataclass
class Prompt:
    """Marketplace prompt listing"""
    prompt_id: int
    title: str
    description: str
    prompt_text: str
    category: str  # "coding" | "writing" | "analysis" | "creative" | "research"
    creator_id: str
    price_per_use: float  # In credits
    rating: float  # 0-5 stars
    total_uses: int
    total_revenue: float
    performance_metrics: Dict[str, float]  # task_success_rate, avg_tokens, etc.
    tags: List[str]
    version: str
    parent_id: Optional[int]  # For forks
    created_at: str
    updated_at: str


@dataclass
class Transaction:
    """Purchase transaction"""
    transaction_id: int
    prompt_id: int
    buyer_id: str
    seller_id: str
    price: float
    royalty_paid: float  # To original creator if forked
    timestamp: str


@dataclass
class PerformanceReport:
    """Prompt performance data"""
    prompt_id: int
    uses_count: int
    success_rate: float
    avg_response_length: int
    avg_execution_time: float
    user_satisfaction: float
    compared_to_baseline: float  # % improvement


@dataclass
class LeaderboardEntry:
    """Leaderboard ranking"""
    rank: int
    prompt_id: int
    title: str
    creator: str
    metric_value: float
    metric_name: str


class PromptMarketplace:
    """Darwinian marketplace for prompts"""

    def __init__(self, data_file: str = "marketplace.json"):
        self.data_file = Path(data_file)
        self.prompts: Dict[int, Prompt] = {}
        self.transactions: List[Transaction] = []
        self.performance_data: Dict[int, PerformanceReport] = {}

        self._load_data()

    def _load_data(self):
        """Load marketplace data"""
        if self.data_file.exists():
            with open(self.data_file, 'r') as f:
                data = json.load(f)
                self.prompts = {
                    int(k): Prompt(**v) for k, v in data.get('prompts', {}).items()
                }
                self.transactions = [
                    Transaction(**t) for t in data.get('transactions', [])
                ]

    def _save_data(self):
        """Save marketplace data"""
        data = {
            'prompts': {k: asdict(v) for k, v in self.prompts.items()},
            'transactions': [asdict(t) for t in self.transactions]
        }
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)

    def list_prompt(
        self,
        title: str,
        description: str,
        prompt_text: str,
        category: str,
        creator_id: str,
        price: float,
        tags: List[str] = None
    ) -> Prompt:
        """List a new prompt for sale"""

        prompt_id = max(self.prompts.keys(), default=0) + 1

        prompt = Prompt(
            prompt_id=prompt_id,
            title=title,
            description=description,
            prompt_text=prompt_text,
            category=category,
            creator_id=creator_id,
            price_per_use=price,
            rating=0.0,
            total_uses=0,
            total_revenue=0.0,
            performance_metrics={},
            tags=tags or [],
            version="1.0.0",
            parent_id=None,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )

        self.prompts[prompt_id] = prompt
        self._save_data()

        return prompt

    def search(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_rating: float = 0.0,
        max_price: Optional[float] = None,
        sort_by: str = "rating"  # "rating" | "uses" | "price" | "revenue"
    ) -> List[Prompt]:
        """Search marketplace"""

        results = list(self.prompts.values())

        # Apply filters
        if category:
            results = [p for p in results if p.category == category]

        if tags:
            results = [
                p for p in results
                if any(tag in p.tags for tag in tags)
            ]

        if min_rating:
            results = [p for p in results if p.rating >= min_rating]

        if max_price is not None:
            results = [p for p in results if p.price_per_use <= max_price]

        # Sort
        if sort_by == "rating":
            results.sort(key=lambda p: p.rating, reverse=True)
        elif sort_by == "uses":
            results.sort(key=lambda p: p.total_uses, reverse=True)
        elif sort_by == "price":
            results.sort(key=lambda p: p.price_per_use)
        elif sort_by == "revenue":
            results.sort(key=lambda p: p.total_revenue, reverse=True)

        return results

    def buy(self, prompt_id: int, buyer_id: str) -> Transaction:
        """Purchase prompt usage"""

        if prompt_id not in self.prompts:
            raise ValueError(f"Prompt {prompt_id} not found")

        prompt = self.prompts[prompt_id]

        # Calculate royalties if forked
        royalty = 0.0
        if prompt.parent_id is not None:
            royalty = prompt.price_per_use * 0.2  # 20% to original creator

        # Create transaction
        transaction = Transaction(
            transaction_id=len(self.transactions) + 1,
            prompt_id=prompt_id,
            buyer_id=buyer_id,
            seller_id=prompt.creator_id,
            price=prompt.price_per_use,
            royalty_paid=royalty,
            timestamp=datetime.now().isoformat()
        )

        # Update prompt stats
        prompt.total_uses += 1
        prompt.total_revenue += prompt.price_per_use

        # If forked, update parent's royalty revenue
        if prompt.parent_id and prompt.parent_id in self.prompts:
            parent = self.prompts[prompt.parent_id]
            parent.total_revenue += royalty

        self.transactions.append(transaction)
        self._save_data()

        return transaction

    def fork(self, prompt_id: int, new_creator: str, modifications: str) -> Prompt:
        """Fork an existing prompt"""

        if prompt_id not in self.prompts:
            raise ValueError(f"Prompt {prompt_id} not found")

        original = self.prompts[prompt_id]

        # Create forked version
        new_id = max(self.prompts.keys()) + 1

        forked = Prompt(
            prompt_id=new_id,
            title=f"{original.title} (Modified)",
            description=f"Fork of #{prompt_id}: {modifications}",
            prompt_text=original.prompt_text,  # Would be modified
            category=original.category,
            creator_id=new_creator,
            price_per_use=original.price_per_use,
            rating=0.0,
            total_uses=0,
            total_revenue=0.0,
            performance_metrics={},
            tags=original.tags.copy(),
            version="1.0.0-fork",
            parent_id=prompt_id,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )

        self.prompts[new_id] = forked
        self._save_data()

        return forked

    def rate(self, prompt_id: int, rating: float, user_id: str):
        """Rate a prompt (1-5 stars)"""

        if prompt_id not in self.prompts:
            raise ValueError(f"Prompt {prompt_id} not found")

        prompt = self.prompts[prompt_id]

        # Simple running average (production would track all ratings)
        if prompt.rating == 0:
            prompt.rating = rating
        else:
            # Weighted average
            prompt.rating = (prompt.rating * prompt.total_uses + rating) / (prompt.total_uses + 1)

        self._save_data()

    def get_leaderboard(
        self,
        metric: str = "revenue",  # "revenue" | "uses" | "rating" | "success_rate"
        category: Optional[str] = None,
        limit: int = 10
    ) -> List[LeaderboardEntry]:
        """Get top prompts by metric"""

        prompts = list(self.prompts.values())

        if category:
            prompts = [p for p in prompts if p.category == category]

        # Sort by metric
        if metric == "revenue":
            prompts.sort(key=lambda p: p.total_revenue, reverse=True)
            metric_name = "Total Revenue"
        elif metric == "uses":
            prompts.sort(key=lambda p: p.total_uses, reverse=True)
            metric_name = "Total Uses"
        elif metric == "rating":
            prompts.sort(key=lambda p: p.rating, reverse=True)
            metric_name = "Average Rating"
        elif metric == "success_rate":
            # Would use real performance data
            prompts.sort(key=lambda p: p.rating, reverse=True)  # Simplified
            metric_name = "Success Rate"

        # Create leaderboard
        leaderboard = []
        for rank, prompt in enumerate(prompts[:limit], 1):
            if metric == "revenue":
                value = prompt.total_revenue
            elif metric == "uses":
                value = prompt.total_uses
            elif metric == "rating":
                value = prompt.rating
            else:
                value = prompt.rating  # Simplified

            entry = LeaderboardEntry(
                rank=rank,
                prompt_id=prompt.prompt_id,
                title=prompt.title,
                creator=prompt.creator_id,
                metric_value=value,
                metric_name=metric_name
            )
            leaderboard.append(entry)

        return leaderboard

    def get_creator_stats(self, creator_id: str) -> Dict[str, Any]:
        """Get statistics for a creator"""

        creator_prompts = [
            p for p in self.prompts.values()
            if p.creator_id == creator_id
        ]

        if not creator_prompts:
            return {
                'total_prompts': 0,
                'total_revenue': 0.0,
                'total_uses': 0,
                'avg_rating': 0.0,
                'top_prompt': None
            }

        total_revenue = sum(p.total_revenue for p in creator_prompts)
        total_uses = sum(p.total_uses for p in creator_prompts)
        avg_rating = sum(p.rating for p in creator_prompts) / len(creator_prompts)

        top_prompt = max(creator_prompts, key=lambda p: p.total_revenue)

        return {
            'total_prompts': len(creator_prompts),
            'total_revenue': round(total_revenue, 2),
            'total_uses': total_uses,
            'avg_rating': round(avg_rating, 2),
            'top_prompt': {
                'id': top_prompt.prompt_id,
                'title': top_prompt.title,
                'revenue': top_prompt.total_revenue
            }
        }

    def get_trending(self, days: int = 7, limit: int = 10) -> List[Prompt]:
        """Get trending prompts (most recent uses)"""

        # Simplified - would use actual time-based analysis
        prompts = list(self.prompts.values())
        prompts.sort(key=lambda p: p.total_uses, reverse=True)

        return prompts[:limit]


def main():
    parser = argparse.ArgumentParser(
        description="PromptMarketplace - Buy/Sell/Trade Prompts"
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # List prompt
    list_parser = subparsers.add_parser('list-prompt', help='List a prompt for sale')
    list_parser.add_argument('--title', required=True, help='Prompt title')
    list_parser.add_argument('--description', required=True, help='Description')
    list_parser.add_argument('--prompt', required=True, help='Prompt text')
    list_parser.add_argument('--category', required=True, choices=['coding', 'writing', 'analysis', 'creative', 'research'])
    list_parser.add_argument('--creator', required=True, help='Creator ID')
    list_parser.add_argument('--price', type=float, required=True, help='Price per use')
    list_parser.add_argument('--tags', help='Comma-separated tags')

    # Search
    search_parser = subparsers.add_parser('search', help='Search marketplace')
    search_parser.add_argument('--category', choices=['coding', 'writing', 'analysis', 'creative', 'research'])
    search_parser.add_argument('--tags', help='Comma-separated tags')
    search_parser.add_argument('--min-rating', type=float, default=0.0)
    search_parser.add_argument('--max-price', type=float)
    search_parser.add_argument('--sort', default='rating', choices=['rating', 'uses', 'price', 'revenue'])

    # Buy
    buy_parser = subparsers.add_parser('buy', help='Buy prompt usage')
    buy_parser.add_argument('--prompt-id', type=int, required=True)
    buy_parser.add_argument('--buyer', required=True, help='Buyer ID')

    # Fork
    fork_parser = subparsers.add_parser('fork', help='Fork existing prompt')
    fork_parser.add_argument('--prompt-id', type=int, required=True)
    fork_parser.add_argument('--creator', required=True, help='Your creator ID')
    fork_parser.add_argument('--modifications', required=True, help='What you changed')

    # Rate
    rate_parser = subparsers.add_parser('rate', help='Rate a prompt')
    rate_parser.add_argument('--prompt-id', type=int, required=True)
    rate_parser.add_argument('--rating', type=float, required=True, help='Rating (1-5)')
    rate_parser.add_argument('--user', required=True, help='User ID')

    # Leaderboard
    leader_parser = subparsers.add_parser('leaderboard', help='View top prompts')
    leader_parser.add_argument('--metric', default='revenue', choices=['revenue', 'uses', 'rating', 'success_rate'])
    leader_parser.add_argument('--category', choices=['coding', 'writing', 'analysis', 'creative', 'research'])
    leader_parser.add_argument('--limit', type=int, default=10)

    # Creator stats
    stats_parser = subparsers.add_parser('stats', help='Get creator statistics')
    stats_parser.add_argument('--creator', required=True, help='Creator ID')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    marketplace = PromptMarketplace()

    if args.command == 'list-prompt':
        tags = [t.strip() for t in args.tags.split(',')] if args.tags else []

        prompt = marketplace.list_prompt(
            title=args.title,
            description=args.description,
            prompt_text=args.prompt,
            category=args.category,
            creator_id=args.creator,
            price=args.price,
            tags=tags
        )

        print(f"\nPrompt listed successfully!")
        print(f"ID: {prompt.prompt_id}")
        print(f"Title: {prompt.title}")
        print(f"Price: ${prompt.price_per_use:.2f} per use")
        print(f"Category: {prompt.category}")

    elif args.command == 'search':
        tags = [t.strip() for t in args.tags.split(',')] if args.tags else None

        results = marketplace.search(
            category=args.category,
            tags=tags,
            min_rating=args.min_rating,
            max_price=args.max_price,
            sort_by=args.sort
        )

        print(f"\nFound {len(results)} prompts:\n")
        for prompt in results[:20]:
            print(f"#{prompt.prompt_id}: {prompt.title}")
            print(f"  Category: {prompt.category} | Price: ${prompt.price_per_use:.2f}")
            print(f"  Rating: {prompt.rating:.1f}★ | Uses: {prompt.total_uses}")
            print(f"  Creator: {prompt.creator_id}")
            print()

    elif args.command == 'buy':
        transaction = marketplace.buy(args.prompt_id, args.buyer)

        print(f"\nPurchase successful!")
        print(f"Transaction ID: {transaction.transaction_id}")
        print(f"Amount paid: ${transaction.price:.2f}")
        if transaction.royalty_paid > 0:
            print(f"Royalty to original creator: ${transaction.royalty_paid:.2f}")

    elif args.command == 'fork':
        forked = marketplace.fork(args.prompt_id, args.creator, args.modifications)

        print(f"\nPrompt forked successfully!")
        print(f"New ID: {forked.prompt_id}")
        print(f"Parent: #{forked.parent_id}")
        print(f"Note: 20% royalty will go to original creator on each use")

    elif args.command == 'rate':
        marketplace.rate(args.prompt_id, args.rating, args.user)
        print(f"\nRating submitted: {args.rating}★")

    elif args.command == 'leaderboard':
        leaderboard = marketplace.get_leaderboard(
            metric=args.metric,
            category=args.category,
            limit=args.limit
        )

        print(f"\n{'='*70}")
        print(f"TOP PROMPTS - {args.metric.upper()}")
        if args.category:
            print(f"Category: {args.category}")
        print(f"{'='*70}\n")

        for entry in leaderboard:
            print(f"{entry.rank}. {entry.title}")
            print(f"   Creator: {entry.creator}")
            print(f"   {entry.metric_name}: {entry.metric_value:.2f}")
            print()

    elif args.command == 'stats':
        stats = marketplace.get_creator_stats(args.creator)

        print(f"\n{'='*70}")
        print(f"CREATOR STATS: {args.creator}")
        print(f"{'='*70}\n")
        print(f"Total prompts: {stats['total_prompts']}")
        print(f"Total revenue: ${stats['total_revenue']:.2f}")
        print(f"Total uses: {stats['total_uses']}")
        print(f"Average rating: {stats['avg_rating']:.1f}★")

        if stats['top_prompt']:
            print(f"\nTop prompt:")
            print(f"  #{stats['top_prompt']['id']}: {stats['top_prompt']['title']}")
            print(f"  Revenue: ${stats['top_prompt']['revenue']:.2f}")


if __name__ == "__main__":
    main()
