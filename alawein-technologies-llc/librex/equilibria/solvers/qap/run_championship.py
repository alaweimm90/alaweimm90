#!/usr/bin/env python
"""
QAP Squad Championship - Main Tournament Runner

Execute with:
    python -m Librex.QAP.run_championship

Or directly:
    python Librex.QAP/run_championship.py
"""

import argparse
from pathlib import Path
from typing import List, Optional

from .benchmarking_suite import QAPSquadChampionship
from .championship_visualizer import ChampionshipVisualizer
from .logging_config import get_logger, setup_logging

# Best known solutions for QAPLIB instances
BEST_KNOWN = {
    # Hadamard instances
    "had12": 1652,
    "had14": 2724,
    "had16": 3720,
    "had18": 5358,
    "had20": 6922,
    # Christofides instances
    "chr12c": 9552,
    "chr15a": 1599,
    "chr15c": 9896,
    "chr18a": 645,
    "chr22a": 6156,
    # Taillard instances
    "tai30a": 1818,
    "tai40a": 3139,
    "tai50a": 4941,
    "tai60a": 7205,
    "tai80a": 13557,
    # NUGU instances
    "nug12": 578,
    "nug14": 1014,
    "nug15": 1150,
    "nug16": 1610,
    "nug17": 2289,
    "nug20": 2570,
    # ROU instances
    "rou12": 235528,
    "rou15": 354210,
    "rou20": 725522,
}


def run_quick_championship(
    data_dir: Path,
    output_dir: Path,
    instances: Optional[List[str]] = None,
) -> None:
    """Run quick championship on subset of instances (for testing)"""
    print("\n" + "=" * 80)
    print("🏆 QAP SQUAD CHAMPIONSHIP - QUICK MODE 🏆")
    print("=" * 80)

    # Default: test a few instances
    if not instances:
        instances = ["had12", "had14", "tai30a", "chr15a", "nug12"]

    logger = get_logger()
    logger.info(f"Quick Championship: {len(instances)} instances")

    championship = QAPSquadChampionship(
        data_dir=data_dir, best_known=BEST_KNOWN, timeout_seconds=300, output_dir=output_dir
    )

    # Run championship
    results = championship.run_championship(
        instances=instances, methods=championship.dispatcher.list_methods("all")
    )

    # Print leaderboard
    championship.print_summary()

    # Save results
    results_path = championship.save_results("quick_championship_results.json")

    # Generate visualizations
    print("\n📊 Generating visualizations...")
    visualizer = ChampionshipVisualizer(
        championship.results, output_dir=output_dir / "visualizations"
    )
    visualizer.generate_all_visualizations()

    print("\n" + "=" * 80)
    print("✅ Championship Complete!")
    print(f"   Results: {results_path}")
    print(f"   Output: {output_dir}")
    print("=" * 80 + "\n")


def run_full_championship(
    data_dir: Path,
    output_dir: Path,
) -> None:
    """Run full championship on all instances"""
    print("\n" + "=" * 80)
    print("🏆 QAP SQUAD CHAMPIONSHIP - FULL MODE 🏆")
    print("=" * 80)

    logger = get_logger()
    instances = list(BEST_KNOWN.keys())
    logger.info(f"Full Championship: {len(instances)} instances")

    championship = QAPSquadChampionship(
        data_dir=data_dir, best_known=BEST_KNOWN, timeout_seconds=600, output_dir=output_dir
    )

    # Run championship
    results = championship.run_championship(
        instances=instances, methods=championship.dispatcher.list_methods("all")
    )

    # Print leaderboard
    championship.print_summary()

    # Save results
    results_path = championship.save_results("full_championship_results.json")

    # Generate visualizations
    print("\n📊 Generating visualizations...")
    visualizer = ChampionshipVisualizer(
        championship.results, output_dir=output_dir / "visualizations"
    )
    visualizer.generate_all_visualizations()

    print("\n" + "=" * 80)
    print("✅ Championship Complete!")
    print(f"   Results: {results_path}")
    print(f"   Output: {output_dir}")
    print("=" * 80 + "\n")


def run_category_championship(
    data_dir: Path,
    output_dir: Path,
    category: str = "micro",
) -> None:
    """Run championship on specific problem size category"""
    print("\n" + "=" * 80)
    print(f"🏆 QAP SQUAD CHAMPIONSHIP - {category.upper()} CATEGORY 🏆")
    print("=" * 80)

    # Map categories to instance names
    categories = {
        "micro": ["had12", "had14", "had16", "chr12c"],
        "small": ["had18", "had20", "chr15a", "chr15c", "tai30a", "nug12", "nug14"],
        "medium": ["chr18a", "chr22a", "tai40a", "tai50a", "nug15", "nug16", "nug17"],
        "large": ["tai60a", "tai80a", "nug20", "rou12", "rou15", "rou20"],
    }

    if category not in categories:
        raise ValueError(f"Unknown category: {category}. Choose from: {list(categories.keys())}")

    instances = categories[category]

    logger = get_logger()
    logger.info(f"Category Championship ({category}): {len(instances)} instances")

    championship = QAPSquadChampionship(
        data_dir=data_dir,
        best_known=BEST_KNOWN,
        timeout_seconds=600,
        output_dir=output_dir / category,
    )

    # Run championship
    results = championship.run_championship(
        instances=instances, methods=championship.dispatcher.list_methods("all")
    )

    # Print leaderboard
    championship.print_summary()

    # Save results
    results_path = championship.save_results(f"{category}_championship_results.json")

    # Generate visualizations
    print(f"\n📊 Generating visualizations for {category}...")
    visualizer = ChampionshipVisualizer(
        championship.results, output_dir=output_dir / category / "visualizations"
    )
    visualizer.generate_all_visualizations()

    print("\n" + "=" * 80)
    print(f"✅ {category.upper()} Championship Complete!")
    print(f"   Results: {results_path}")
    print(f"   Output: {output_dir / category}")
    print("=" * 80 + "\n")


def print_usage_guide():
    """Print guide for running championships"""
    guide = """
╔════════════════════════════════════════════════════════════════════════════╗
║              QAP SQUAD CHAMPIONSHIP RUNNER - USER GUIDE                   ║
╚════════════════════════════════════════════════════════════════════════════╝

Three ways to run championships:

1️⃣  QUICK MODE (5-10 instances, ~5-10 minutes)
   └─ Good for testing and quick comparisons
   └─ python -m Librex.QAP.run_championship --quick

2️⃣  CATEGORY MODE (By problem size)
   └─ MICRO:   n < 20 (small problems, ~5-10 min)
   └─ SMALL:   20 ≤ n < 40 (medium, ~20-30 min)
   └─ MEDIUM:  40 ≤ n < 60 (harder, ~30-45 min)
   └─ LARGE:   n ≥ 60 (large-scale, ~45-60 min)

   └─ python -m Librex.QAP.run_championship --category micro
   └─ python -m Librex.QAP.run_championship --category large

3️⃣  FULL MODE (All ~25 instances, ~60-90 minutes)
   └─ Comprehensive championship across all problem types
   └─ python -m Librex.QAP.run_championship --full

═══════════════════════════════════════════════════════════════════════════

OUTPUT FILES GENERATED:

📊 Championship Results
   └─ championship_results.json (full data)

📈 Visualizations (5 charts)
   ├─ 01_gap_heatmap.png (methods × instances)
   ├─ 02_efficiency_frontier.png (time vs quality)
   ├─ 03_win_rates.png (win rate by method)
   ├─ 04_performance_by_size.png (trajectory)
   └─ 05_consistency.png (quality vs consistency)

📋 Leaderboard (printed to console)
   └─ Full rankings with gap, wins, points, efficiency

═══════════════════════════════════════════════════════════════════════════

EXAMPLE WORKFLOW:

1. Start with quick championship to test:
   $ python -m Librex.QAP.run_championship --quick

2. If good, run by category:
   $ python -m Librex.QAP.run_championship --category micro
   $ python -m Librex.QAP.run_championship --category small

3. Finally, run full championship:
   $ python -m Librex.QAP.run_championship --full

═══════════════════════════════════════════════════════════════════════════

INTERPRETING RESULTS:

🥇 LEADERBOARD
   • Rank: Position (1st, 2nd, 3rd, etc.)
   • AVG_GAP: Average optimality gap (lower = better)
   • WINS: Number of instances where this method was best
   • POINTS: Total championship points
   • CONSISTENCY: How reliable is this method (1.0 = perfect)
   • EFFICIENCY: Quality per second (higher = better)

📊 HEATMAP (Green = Good, Red = Bad)
   • Shows gap for each method-instance pair
   • Helps identify which methods work best for specific problems

📈 EFFICIENCY FRONTIER (Green dashed line)
   • Shows Pareto-optimal methods
   • Methods on frontier = best bang-for-buck
   • Methods above frontier = dominated by others

🏆 WIN RATES
   • Percentage of instances where each method was best
   • Cumulative achievement across all test cases

═══════════════════════════════════════════════════════════════════════════
"""
    print(guide)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="🏆 QAP Squad Championship - Comprehensive Benchmarking System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m Librex.QAP.run_championship --quick
  python -m Librex.QAP.run_championship --category micro
  python -m Librex.QAP.run_championship --full
        """,
    )

    parser.add_argument(
        "--quick", action="store_true", help="Run quick championship on ~5 instances"
    )

    parser.add_argument(
        "--category",
        choices=["micro", "small", "medium", "large"],
        help="Run championship on specific problem size category",
    )

    parser.add_argument(
        "--full", action="store_true", help="Run full championship on all ~25 instances"
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default="./data/qaplib",
        help="Path to QAPLIB data directory (default: ./data/qaplib)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default="./championship_results",
        help="Output directory for results (default: ./championship_results)",
    )

    parser.add_argument("--guide", action="store_true", help="Print usage guide")

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    # Handle guide request
    if args.guide:
        print_usage_guide()
        return

    # Ensure at least one mode is selected
    if not any([args.quick, args.category, args.full]):
        print_usage_guide()
        parser.print_help()
        return

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Run selected mode
    if args.quick:
        run_quick_championship(args.data_dir, args.output_dir)

    elif args.category:
        run_category_championship(args.data_dir, args.output_dir, args.category)

    elif args.full:
        run_full_championship(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
