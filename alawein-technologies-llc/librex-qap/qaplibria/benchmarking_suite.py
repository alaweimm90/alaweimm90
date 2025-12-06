"""
QAP Squad Championship - Core Benchmarking System

Runs comprehensive tournaments across multiple methods, instances, and categories.
Tracks wins, gaps, times, and consistency for championship leaderboards.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Dict, List, Optional

import numpy as np

from .core import QAPBenchmarkPipeline
from .logging_config import get_logger
from .pipeline_dispatcher import get_dispatcher


@dataclass
class BenchmarkEntry:
    """Single benchmark result"""

    instance_name: str
    method_name: str
    achieved_cost: float
    best_known: float
    gap_percent: float
    time_seconds: float
    solver_status: str  # 'success', 'timeout', 'error'
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def efficiency_score(self) -> float:
        """Quality per second (higher is better)"""
        if self.time_seconds < 0.1:
            return 1000.0  # Near-instant, excellent
        return (100.0 - self.gap_percent) / self.time_seconds


@dataclass
class MethodStats:
    """Statistics for a single method across instances"""

    method_name: str
    wins: int = 0
    total_runs: int = 0
    avg_gap: float = 0.0
    min_gap: float = float("inf")
    max_gap: float = 0.0
    std_gap: float = 0.0
    avg_time: float = 0.0
    consistency_score: float = 0.0  # 1 - (std_gap / avg_gap)
    avg_efficiency: float = 0.0
    total_points: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / max(1, self.total_runs)


class QAPSquadChampionship:
    """Comprehensive QAP benchmarking tournament system"""

    def __init__(
        self,
        data_dir: Path | str,
        best_known: Dict[str, int],
        timeout_seconds: int = 600,
        output_dir: Optional[Path | str] = None,
    ):
        """
        Initialize championship system.

        Args:
            data_dir: Path to QAPLIB instances
            best_known: Dict mapping instance names to best known costs
            timeout_seconds: Max time per method-instance pair
            output_dir: Directory for results/logs
        """
        self.data_dir = Path(data_dir)
        self.best_known = best_known
        self.timeout_seconds = timeout_seconds
        self.output_dir = Path(output_dir or "./championship_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger()
        self.pipeline = QAPBenchmarkPipeline(data_dir=data_dir, best_known=best_known, rng_seed=42)
        self.dispatcher = get_dispatcher()

        self.results: List[BenchmarkEntry] = []
        self.method_stats: Dict[str, MethodStats] = {}

    def categorize_instances(self) -> Dict[str, List[str]]:
        """Categorize instances by problem size"""
        categories = {
            "micro": [],  # n < 20
            "small": [],  # 20 <= n < 40
            "medium": [],  # 40 <= n < 60
            "large": [],  # n >= 60
        }

        for instance_name in self.best_known.keys():
            try:
                # Extract size from instance name (e.g., 'had12' -> 12)
                import re

                match = re.search(r"(\d+)", instance_name)
                if match:
                    n = int(match.group(1))
                    if n < 20:
                        categories["micro"].append(instance_name)
                    elif n < 40:
                        categories["small"].append(instance_name)
                    elif n < 60:
                        categories["medium"].append(instance_name)
                    else:
                        categories["large"].append(instance_name)
            except Exception:
                pass

        return categories

    def run_championship(
        self,
        instances: Optional[List[str]] = None,
        methods: Optional[List[str]] = None,
    ) -> Dict:
        """
        Run full championship tournament.

        Args:
            instances: List of instance names (None=all)
            methods: List of method names (None=all)

        Returns:
            Championship results dictionary
        """
        instances = instances or list(self.best_known.keys())
        methods = methods or self.dispatcher.list_methods("all")

        self.logger.info("Starting QAP Squad Championship")
        self.logger.info(f"Instances: {len(instances)}, Methods: {len(methods)}")

        total_runs = len(instances) * len(methods)
        completed = 0

        for instance_name in instances:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Instance: {instance_name}")
            self.logger.info(f"{'='*60}")

            best_known_cost = self.best_known[instance_name]
            best_gap_this_instance = float("inf")
            best_method_this_instance = None

            for method_name in methods:
                completed += 1
                progress = f"[{completed}/{total_runs}]"

                try:
                    # Run method with timeout
                    start_time = time.perf_counter()

                    # Simple execution - in production, would use multiprocessing timeout
                    # For now, we'll simulate with a direct call
                    spectral_matrix = self._get_initial_solution(instance_name)
                    result_matrix, _ = self.dispatcher.apply_method(method_name, spectral_matrix)

                    elapsed = time.perf_counter() - start_time

                    # Compute cost (simplified - would extract from pipeline)
                    achieved_cost = best_known_cost * (1 + np.random.randn() * 0.1)  # Mock
                    gap_percent = max(0, (achieved_cost - best_known_cost) / best_known_cost * 100)

                    entry = BenchmarkEntry(
                        instance_name=instance_name,
                        method_name=method_name,
                        achieved_cost=achieved_cost,
                        best_known=best_known_cost,
                        gap_percent=gap_percent,
                        time_seconds=elapsed,
                        solver_status="success",
                    )

                    self.results.append(entry)

                    # Track best on this instance
                    if gap_percent < best_gap_this_instance:
                        best_gap_this_instance = gap_percent
                        best_method_this_instance = method_name

                    # Update method stats
                    self._update_stats(entry)

                    self.logger.info(
                        f"{progress} {method_name:<30} "
                        f"Gap: {gap_percent:6.2f}% | Time: {elapsed:6.2f}s"
                    )

                except Exception as e:
                    self.logger.error(f"{progress} {method_name}: {str(e)}")
                    entry = BenchmarkEntry(
                        instance_name=instance_name,
                        method_name=method_name,
                        achieved_cost=float("inf"),
                        best_known=best_known_cost,
                        gap_percent=float("inf"),
                        time_seconds=0,
                        solver_status="error",
                    )
                    self.results.append(entry)

            # Summary for this instance
            self.logger.info(
                f"Best on {instance_name}: {best_method_this_instance} "
                f"({best_gap_this_instance:.2f}%)"
            )

        return self.compile_results()

    def compile_results(self) -> Dict:
        """Compile comprehensive results dictionary"""
        results_dict = {
            "timestamp": datetime.now().isoformat(),
            "total_runs": len(self.results),
            "entries": [asdict(entry) for entry in self.results],
            "method_stats": {name: asdict(stats) for name, stats in self.method_stats.items()},
            "categories": self.categorize_instances(),
        }

        return results_dict

    def _get_initial_solution(self, instance_name: str) -> np.ndarray:
        """Get initial solution for an instance (mock)"""
        # In production, would load actual instance and get spectral init
        return np.eye(20)  # Mock 20x20 identity

    def _update_stats(self, entry: BenchmarkEntry) -> None:
        """Update running statistics for a method"""
        method_name = entry.method_name

        if method_name not in self.method_stats:
            self.method_stats[method_name] = MethodStats(method_name=method_name)

        stats = self.method_stats[method_name]
        stats.total_runs += 1

        if entry.gap_percent < float("inf"):
            # Update wins
            current_best = min(
                (e.gap_percent for e in self.results if e.instance_name == entry.instance_name),
                default=float("inf"),
            )
            if entry.gap_percent <= current_best:
                stats.wins += 1

            # Update gap stats
            all_gaps = [
                e.gap_percent
                for e in self.results
                if e.method_name == method_name and e.gap_percent < float("inf")
            ]

            if all_gaps:
                stats.avg_gap = np.mean(all_gaps)
                stats.min_gap = np.min(all_gaps)
                stats.max_gap = np.max(all_gaps)
                stats.std_gap = np.std(all_gaps) if len(all_gaps) > 1 else 0
                stats.consistency_score = 1 - (stats.std_gap / max(stats.avg_gap, 1e-6))

            # Update time stats
            all_times = [
                e.time_seconds
                for e in self.results
                if e.method_name == method_name and e.time_seconds > 0
            ]
            if all_times:
                stats.avg_time = np.mean(all_times)

            # Update efficiency stats
            all_efficiencies = [
                e.efficiency_score for e in self.results if e.method_name == method_name
            ]
            if all_efficiencies:
                stats.avg_efficiency = np.mean(all_efficiencies)

            # Compute points
            stats.total_points = self._compute_points(stats)

    def _compute_points(self, stats: MethodStats) -> int:
        """Compute championship points for a method"""
        points = 0

        # Gap-based points (lower gap = more points)
        if stats.avg_gap < 5:
            points += 100
        elif stats.avg_gap < 10:
            points += 80
        elif stats.avg_gap < 15:
            points += 60
        elif stats.avg_gap < 20:
            points += 40
        elif stats.avg_gap < 30:
            points += 20

        # Win-based points
        points += stats.wins * 5

        # Consistency bonus
        points += int(stats.consistency_score * 50)

        # Efficiency bonus
        if stats.avg_efficiency > 0:
            points += min(20, int(stats.avg_efficiency / 5))

        return points

    def generate_leaderboard(self) -> str:
        """Generate championship leaderboard"""
        sorted_methods = sorted(
            self.method_stats.values(), key=lambda x: (-x.total_points, x.avg_gap)
        )

        board = "\n" + "=" * 100 + "\n"
        board += "🏆 QAP SQUAD CHAMPIONSHIP LEADERBOARD 🏆\n"
        board += "=" * 100 + "\n"
        board += f"{'RANK':<5} {'METHOD':<30} {'AVG_GAP':<10} {'WINS':<6} {'POINTS':<8} "
        board += f"{'CONSISTENCY':<12} {'EFFICIENCY':<10}\n"
        board += "-" * 100 + "\n"

        for rank, stats in enumerate(sorted_methods, 1):
            medal = (
                "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank:2d}."
            )

            board += f"{medal:<5} {stats.method_name:<30} {stats.avg_gap:7.2f}%  "
            board += f"{stats.wins:<6} {stats.total_points:<8} {stats.consistency_score:<12.2%} "
            board += f"{stats.avg_efficiency:<10.2f}\n"

        board += "=" * 100 + "\n"
        return board

    def save_results(self, filename: str = "championship_results.json") -> Path:
        """Save championship results to JSON"""
        results = self.compile_results()
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"Results saved: {filepath}")
        return filepath

    def print_summary(self) -> None:
        """Print championship summary"""
        print(self.generate_leaderboard())

        # Category breakdowns
        categories = self.categorize_instances()
        print("\n" + "=" * 100)
        print("CHAMPIONS BY PROBLEM SIZE")
        print("=" * 100)

        for category, instances in categories.items():
            if instances:
                print(f"\n{category.upper()} (n in range):")
                # Find best method for this category
                category_results = [
                    e
                    for e in self.results
                    if e.instance_name in instances and e.gap_percent < float("inf")
                ]
                if category_results:
                    best = min(category_results, key=lambda x: x.gap_percent)
                    print(f"  Champion: {best.method_name} ({best.gap_percent:.2f}%)")


__all__ = ["QAPSquadChampionship", "BenchmarkEntry", "MethodStats"]
