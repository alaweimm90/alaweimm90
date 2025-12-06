#!/usr/bin/env python3
"""
FailureDB - Failure Futures Market

Prediction market for research failures. Learn from what didn't work.
- Submit failed experiments with details
- Bet on which ideas will fail
- Earn from accurate predictions
- Access failure analytics and lessons learned
- Avoid repeating failed approaches

Usage:
    python failuredb.py submit-failure --title "Cold fusion attempt" --reason "Thermodynamics" --evidence "..."
    python failuredb.py create-market --idea "Room temperature superconductors" --deadline 2025-12-31
    python failuredb.py bet --market-id 42 --outcome FAIL --amount 100
    python failuredb.py search --domain "physics" --reason "thermodynamics"
    python failuredb.py analytics --domain "ML" --timeframe "2024"
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional


@dataclass
class Failure:
    """Failed experiment record"""
    failure_id: int
    title: str
    domain: str
    hypothesis: str
    approach: str
    failure_reason: str
    evidence: str
    cost_wasted: float  # USD
    time_wasted: int  # Days
    lessons_learned: List[str]
    replication_attempts: int
    submitter_id: str
    verified: bool
    upvotes: int
    tags: List[str]
    submitted_at: str


@dataclass
class PredictionMarket:
    """Prediction market for idea failure"""
    market_id: int
    idea_title: str
    idea_description: str
    domain: str
    created_by: str
    deadline: str  # ISO date
    total_pool: float  # Total $ in market
    fail_probability: float  # Current market probability
    positions: Dict[str, Dict[str, float]]  # user_id -> {FAIL: amount, SUCCEED: amount}
    resolved: bool
    actual_outcome: Optional[str]  # "FAIL" | "SUCCEED" | None
    created_at: str


@dataclass
class Bet:
    """Individual bet on market"""
    bet_id: int
    market_id: int
    user_id: str
    outcome: str  # "FAIL" | "SUCCEED"
    amount: float
    probability_at_bet: float
    timestamp: str


@dataclass
class FailureAnalytics:
    """Analytics on failure patterns"""
    domain: str
    total_failures: int
    common_reasons: List[tuple[str, int]]
    avg_cost_wasted: float
    avg_time_wasted: int
    top_lessons: List[str]
    failure_rate_by_approach: Dict[str, float]


class FailureDB:
    """Database of failed experiments and prediction markets"""

    def __init__(self, data_file: str = "failuredb.json"):
        self.data_file = Path(data_file)
        self.failures: Dict[int, Failure] = {}
        self.markets: Dict[int, PredictionMarket] = {}
        self.bets: List[Bet] = []

        self._load_data()

    def _load_data(self):
        """Load database"""
        if self.data_file.exists():
            with open(self.data_file, 'r') as f:
                data = json.load(f)
                self.failures = {
                    int(k): Failure(**v) for k, v in data.get('failures', {}).items()
                }
                self.markets = {
                    int(k): PredictionMarket(**v) for k, v in data.get('markets', {}).items()
                }
                self.bets = [Bet(**b) for b in data.get('bets', [])]

    def _save_data(self):
        """Save database"""
        data = {
            'failures': {k: asdict(v) for k, v in self.failures.items()},
            'markets': {k: asdict(v) for k, v in self.markets.items()},
            'bets': [asdict(b) for b in self.bets]
        }
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)

    def submit_failure(
        self,
        title: str,
        domain: str,
        hypothesis: str,
        approach: str,
        failure_reason: str,
        evidence: str,
        cost_wasted: float,
        time_wasted: int,
        lessons_learned: List[str],
        submitter_id: str,
        tags: List[str] = None
    ) -> Failure:
        """Submit a failed experiment"""

        failure_id = max(self.failures.keys(), default=0) + 1

        failure = Failure(
            failure_id=failure_id,
            title=title,
            domain=domain,
            hypothesis=hypothesis,
            approach=approach,
            failure_reason=failure_reason,
            evidence=evidence,
            cost_wasted=cost_wasted,
            time_wasted=time_wasted,
            lessons_learned=lessons_learned,
            replication_attempts=0,
            submitter_id=submitter_id,
            verified=False,
            upvotes=0,
            tags=tags or [],
            submitted_at=datetime.now().isoformat()
        )

        self.failures[failure_id] = failure
        self._save_data()

        return failure

    def search_failures(
        self,
        domain: Optional[str] = None,
        reason: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_cost: Optional[float] = None,
        verified_only: bool = False
    ) -> List[Failure]:
        """Search failure database"""

        results = list(self.failures.values())

        if domain:
            results = [f for f in results if f.domain.lower() == domain.lower()]

        if reason:
            results = [
                f for f in results
                if reason.lower() in f.failure_reason.lower()
            ]

        if tags:
            results = [
                f for f in results
                if any(tag in f.tags for tag in tags)
            ]

        if min_cost is not None:
            results = [f for f in results if f.cost_wasted >= min_cost]

        if verified_only:
            results = [f for f in results if f.verified]

        # Sort by upvotes
        results.sort(key=lambda f: f.upvotes, reverse=True)

        return results

    def create_market(
        self,
        idea_title: str,
        idea_description: str,
        domain: str,
        creator_id: str,
        deadline: str
    ) -> PredictionMarket:
        """Create prediction market for idea failure"""

        market_id = max(self.markets.keys(), default=0) + 1

        market = PredictionMarket(
            market_id=market_id,
            idea_title=idea_title,
            idea_description=idea_description,
            domain=domain,
            created_by=creator_id,
            deadline=deadline,
            total_pool=0.0,
            fail_probability=0.5,  # Start at 50/50
            positions=defaultdict(lambda: {"FAIL": 0.0, "SUCCEED": 0.0}),
            resolved=False,
            actual_outcome=None,
            created_at=datetime.now().isoformat()
        )

        self.markets[market_id] = market
        self._save_data()

        return market

    def place_bet(
        self,
        market_id: int,
        user_id: str,
        outcome: str,
        amount: float
    ) -> Bet:
        """Place bet on market outcome"""

        if market_id not in self.markets:
            raise ValueError(f"Market {market_id} not found")

        market = self.markets[market_id]

        if market.resolved:
            raise ValueError("Market already resolved")

        # Create bet
        bet = Bet(
            bet_id=len(self.bets) + 1,
            market_id=market_id,
            user_id=user_id,
            outcome=outcome,
            amount=amount,
            probability_at_bet=market.fail_probability,
            timestamp=datetime.now().isoformat()
        )

        self.bets.append(bet)

        # Update market
        market.total_pool += amount

        # Initialize user position if doesn't exist
        if user_id not in market.positions:
            market.positions[user_id] = {"FAIL": 0.0, "SUCCEED": 0.0}

        market.positions[user_id][outcome] = market.positions[user_id].get(outcome, 0) + amount

        # Update probability (simplified market maker)
        fail_total = sum(
            pos["FAIL"] for pos in market.positions.values()
        )
        succeed_total = sum(
            pos["SUCCEED"] for pos in market.positions.values()
        )
        total = fail_total + succeed_total

        if total > 0:
            market.fail_probability = fail_total / total

        self._save_data()

        return bet

    def resolve_market(
        self,
        market_id: int,
        actual_outcome: str,
        resolver_id: str
    ) -> Dict[str, float]:
        """Resolve market and calculate payouts"""

        if market_id not in self.markets:
            raise ValueError(f"Market {market_id} not found")

        market = self.markets[market_id]

        if market.resolved:
            raise ValueError("Market already resolved")

        market.resolved = True
        market.actual_outcome = actual_outcome

        # Calculate payouts
        payouts = {}

        for user_id, positions in market.positions.items():
            bet_amount = positions.get(actual_outcome, 0)

            if bet_amount > 0:
                # Winner gets proportional share of total pool
                total_winning_bets = sum(
                    pos.get(actual_outcome, 0)
                    for pos in market.positions.values()
                )

                if total_winning_bets > 0:
                    payout = (bet_amount / total_winning_bets) * market.total_pool
                    payouts[user_id] = payout

        self._save_data()

        return payouts

    def get_analytics(
        self,
        domain: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> FailureAnalytics:
        """Get failure analytics"""

        failures = list(self.failures.values())

        if domain:
            failures = [f for f in failures if f.domain.lower() == domain.lower()]

        if timeframe:
            # Filter by year
            failures = [
                f for f in failures
                if timeframe in f.submitted_at
            ]

        if not failures:
            return FailureAnalytics(
                domain=domain or "all",
                total_failures=0,
                common_reasons=[],
                avg_cost_wasted=0.0,
                avg_time_wasted=0,
                top_lessons=[],
                failure_rate_by_approach={}
            )

        # Calculate analytics
        reasons = Counter(f.failure_reason for f in failures)
        avg_cost = sum(f.cost_wasted for f in failures) / len(failures)
        avg_time = sum(f.time_wasted for f in failures) / len(failures)

        # Collect all lessons
        all_lessons = []
        for f in failures:
            all_lessons.extend(f.lessons_learned)
        lesson_counts = Counter(all_lessons)
        top_lessons = [lesson for lesson, _ in lesson_counts.most_common(10)]

        # Failure rate by approach (simplified)
        approach_counts = Counter(f.approach for f in failures)
        failure_rate = {
            approach: count / len(failures)
            for approach, count in approach_counts.items()
        }

        return FailureAnalytics(
            domain=domain or "all",
            total_failures=len(failures),
            common_reasons=reasons.most_common(10),
            avg_cost_wasted=round(avg_cost, 2),
            avg_time_wasted=int(avg_time),
            top_lessons=top_lessons,
            failure_rate_by_approach=failure_rate
        )

    def get_leaderboard(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top predictors"""

        # Calculate user performance
        user_stats = defaultdict(lambda: {"correct": 0, "total": 0, "profit": 0.0})

        resolved_markets = [m for m in self.markets.values() if m.resolved]

        for market in resolved_markets:
            # Find bets for this market
            market_bets = [b for b in self.bets if b.market_id == market.market_id]

            for bet in market_bets:
                user_stats[bet.user_id]["total"] += 1

                if bet.outcome == market.actual_outcome:
                    user_stats[bet.user_id]["correct"] += 1

        # Calculate accuracy
        leaderboard = []
        for user_id, stats in user_stats.items():
            if stats["total"] > 0:
                accuracy = stats["correct"] / stats["total"]
                leaderboard.append({
                    "user_id": user_id,
                    "accuracy": round(accuracy, 3),
                    "total_bets": stats["total"],
                    "correct_bets": stats["correct"],
                    "profit": stats["profit"]
                })

        leaderboard.sort(key=lambda x: (x["accuracy"], x["total_bets"]), reverse=True)

        return leaderboard[:limit]


def main():
    parser = argparse.ArgumentParser(
        description="FailureDB - Failure Futures Market"
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Submit failure
    submit_parser = subparsers.add_parser('submit-failure', help='Submit failed experiment')
    submit_parser.add_argument('--title', required=True, help='Failure title')
    submit_parser.add_argument('--domain', required=True, help='Research domain')
    submit_parser.add_argument('--hypothesis', required=True, help='What you were testing')
    submit_parser.add_argument('--approach', required=True, help='Your approach')
    submit_parser.add_argument('--reason', required=True, help='Why it failed')
    submit_parser.add_argument('--evidence', required=True, help='Evidence of failure')
    submit_parser.add_argument('--cost', type=float, default=0, help='$ wasted')
    submit_parser.add_argument('--time', type=int, default=0, help='Days wasted')
    submit_parser.add_argument('--lessons', required=True, help='Comma-separated lessons learned')
    submit_parser.add_argument('--submitter', required=True, help='Your user ID')
    submit_parser.add_argument('--tags', help='Comma-separated tags')

    # Search failures
    search_parser = subparsers.add_parser('search', help='Search failure database')
    search_parser.add_argument('--domain', help='Filter by domain')
    search_parser.add_argument('--reason', help='Search in failure reasons')
    search_parser.add_argument('--tags', help='Comma-separated tags')
    search_parser.add_argument('--min-cost', type=float, help='Minimum cost wasted')
    search_parser.add_argument('--verified-only', action='store_true')

    # Create market
    market_parser = subparsers.add_parser('create-market', help='Create prediction market')
    market_parser.add_argument('--idea', required=True, help='Idea title')
    market_parser.add_argument('--description', required=True, help='Idea description')
    market_parser.add_argument('--domain', required=True, help='Research domain')
    market_parser.add_argument('--creator', required=True, help='Your user ID')
    market_parser.add_argument('--deadline', required=True, help='Deadline (YYYY-MM-DD)')

    # Place bet
    bet_parser = subparsers.add_parser('bet', help='Place bet on market')
    bet_parser.add_argument('--market-id', type=int, required=True)
    bet_parser.add_argument('--outcome', required=True, choices=['FAIL', 'SUCCEED'])
    bet_parser.add_argument('--amount', type=float, required=True, help='Bet amount ($)')
    bet_parser.add_argument('--user', required=True, help='Your user ID')

    # View markets
    markets_parser = subparsers.add_parser('markets', help='View active markets')
    markets_parser.add_argument('--domain', help='Filter by domain')

    # Analytics
    analytics_parser = subparsers.add_parser('analytics', help='Failure analytics')
    analytics_parser.add_argument('--domain', help='Filter by domain')
    analytics_parser.add_argument('--timeframe', help='Year (e.g., 2024)')

    # Leaderboard
    leader_parser = subparsers.add_parser('leaderboard', help='Top predictors')
    leader_parser.add_argument('--limit', type=int, default=10)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    db = FailureDB()

    if args.command == 'submit-failure':
        lessons = [l.strip() for l in args.lessons.split(',')]
        tags = [t.strip() for t in args.tags.split(',')] if args.tags else []

        failure = db.submit_failure(
            title=args.title,
            domain=args.domain,
            hypothesis=args.hypothesis,
            approach=args.approach,
            failure_reason=args.reason,
            evidence=args.evidence,
            cost_wasted=args.cost,
            time_wasted=args.time,
            lessons_learned=lessons,
            submitter_id=args.submitter,
            tags=tags
        )

        print(f"\nFailure submitted successfully!")
        print(f"ID: {failure.failure_id}")
        print(f"Title: {failure.title}")
        print(f"Cost wasted: ${failure.cost_wasted:,.2f}")
        print(f"Time wasted: {failure.time_wasted} days")
        print(f"\nLessons learned:")
        for lesson in failure.lessons_learned:
            print(f"  - {lesson}")

    elif args.command == 'search':
        tags = [t.strip() for t in args.tags.split(',')] if args.tags else None

        results = db.search_failures(
            domain=args.domain,
            reason=args.reason,
            tags=tags,
            min_cost=args.min_cost,
            verified_only=args.verified_only
        )

        print(f"\nFound {len(results)} failures:\n")

        for failure in results[:20]:
            print(f"#{failure.failure_id}: {failure.title}")
            print(f"  Domain: {failure.domain}")
            print(f"  Reason: {failure.failure_reason}")
            print(f"  Cost: ${failure.cost_wasted:,.2f} | Time: {failure.time_wasted} days")
            print(f"  Upvotes: {failure.upvotes} | Verified: {failure.verified}")
            print(f"  Lessons: {', '.join(failure.lessons_learned[:2])}")
            print()

    elif args.command == 'create-market':
        market = db.create_market(
            idea_title=args.idea,
            idea_description=args.description,
            domain=args.domain,
            creator_id=args.creator,
            deadline=args.deadline
        )

        print(f"\nPrediction market created!")
        print(f"Market ID: {market.market_id}")
        print(f"Idea: {market.idea_title}")
        print(f"Deadline: {market.deadline}")
        print(f"Initial probability of failure: {market.fail_probability:.1%}")

    elif args.command == 'bet':
        bet = db.place_bet(
            market_id=args.market_id,
            user_id=args.user,
            outcome=args.outcome,
            amount=args.amount
        )

        market = db.markets[args.market_id]

        print(f"\nBet placed successfully!")
        print(f"Bet ID: {bet.bet_id}")
        print(f"Market: {market.idea_title}")
        print(f"Your bet: ${bet.amount:.2f} on {bet.outcome}")
        print(f"Probability at bet: {bet.probability_at_bet:.1%}")
        print(f"\nUpdated market probability:")
        print(f"  FAIL: {market.fail_probability:.1%}")
        print(f"  SUCCEED: {(1 - market.fail_probability):.1%}")
        print(f"Total pool: ${market.total_pool:.2f}")

    elif args.command == 'markets':
        markets = [
            m for m in db.markets.values()
            if not m.resolved
        ]

        if args.domain:
            markets = [m for m in markets if m.domain.lower() == args.domain.lower()]

        print(f"\n{'='*70}")
        print(f"ACTIVE PREDICTION MARKETS")
        print(f"{'='*70}\n")

        for market in markets:
            print(f"Market #{market.market_id}: {market.idea_title}")
            print(f"  Domain: {market.domain}")
            print(f"  Deadline: {market.deadline}")
            print(f"  Failure probability: {market.fail_probability:.1%}")
            print(f"  Total pool: ${market.total_pool:.2f}")
            print()

    elif args.command == 'analytics':
        analytics = db.get_analytics(
            domain=args.domain,
            timeframe=args.timeframe
        )

        print(f"\n{'='*70}")
        print(f"FAILURE ANALYTICS - {analytics.domain.upper()}")
        print(f"{'='*70}\n")

        print(f"Total failures: {analytics.total_failures}")
        print(f"Average cost wasted: ${analytics.avg_cost_wasted:,.2f}")
        print(f"Average time wasted: {analytics.avg_time_wasted} days")

        print(f"\nMost common failure reasons:")
        for reason, count in analytics.common_reasons[:5]:
            pct = (count / analytics.total_failures * 100)
            print(f"  {reason:40s} {count:3d} ({pct:.1f}%)")

        print(f"\nTop lessons learned:")
        for i, lesson in enumerate(analytics.top_lessons[:5], 1):
            print(f"  {i}. {lesson}")

        if analytics.failure_rate_by_approach:
            print(f"\nFailure rate by approach:")
            for approach, rate in sorted(
                analytics.failure_rate_by_approach.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]:
                print(f"  {approach:40s} {rate:.1%}")

    elif args.command == 'leaderboard':
        leaderboard = db.get_leaderboard(args.limit)

        print(f"\n{'='*70}")
        print(f"TOP PREDICTORS")
        print(f"{'='*70}\n")

        for i, entry in enumerate(leaderboard, 1):
            print(f"{i}. {entry['user_id']}")
            print(f"   Accuracy: {entry['accuracy']:.1%} ({entry['correct_bets']}/{entry['total_bets']})")
            print()


if __name__ == "__main__":
    main()
