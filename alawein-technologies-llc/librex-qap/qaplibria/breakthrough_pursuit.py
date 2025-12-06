"""
Breakthrough Pursuit System - Continuous Optimization Loop

Automatically runs benchmarks, analyzes results, identifies opportunities,
and pursues breakthroughs iteratively.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional

from .benchmarking_suite import QAPSquadChampionship
from .logging_config import get_logger


@dataclass
class BreakthroughTier:
    """Definition of a breakthrough tier"""

    name: str
    gap_target: float
    consistency_target: float
    win_rate_target: float
    time_target: Optional[float] = None
    description: str = ""


@dataclass
class BreakthroughMilestone:
    """Weekly breakthrough tracking"""

    week: int
    date: str
    tier_target: str
    current_avg_gap: float = float("inf")
    current_consistency: float = 0.0
    current_win_rate: float = 0.0
    breakthrough_achieved: bool = False
    progress_notes: str = ""
    next_week_focus: List[str] = field(default_factory=list)


class BreakthroughPursuitSystem:
    """Continuous system for pursuing breakthrough achievements"""

    # Define breakthrough tiers
    TIERS = {
        "entry": BreakthroughTier(
            name="Entry Breakthrough",
            gap_target=15.0,
            consistency_target=0.80,
            win_rate_target=0.20,
            description="Gap < 15% on all instances, Consistency > 80%",
        ),
        "efficiency": BreakthroughTier(
            name="Efficiency Breakthrough",
            gap_target=12.0,
            consistency_target=0.85,
            win_rate_target=0.40,
            time_target=60.0,
            description="Gap < 12%, Time < 60s, On Pareto frontier",
        ),
        "scale": BreakthroughTier(
            name="Scale Breakthrough",
            gap_target=20.0,
            consistency_target=0.75,
            win_rate_target=0.50,
            description="Gap < 20% on large (n≥60), Scales efficiently",
        ),
        "novel": BreakthroughTier(
            name="Novel Breakthrough",
            gap_target=10.0,
            consistency_target=0.90,
            win_rate_target=0.60,
            description="Gap < 10%, Beats all baselines, Novel insight",
        ),
        "championship": BreakthroughTier(
            name="Championship Breakthrough",
            gap_target=8.0,
            consistency_target=0.95,
            win_rate_target=0.70,
            description="Gap < 8%, Consistency > 95%, Win rate > 70%",
        ),
        "extreme": BreakthroughTier(
            name="Extreme Breakthrough",
            gap_target=5.0,
            consistency_target=0.98,
            win_rate_target=0.80,
            description="Gap < 5%, Near-optimal, Revolutionary",
        ),
    }

    def __init__(
        self,
        data_dir: Path | str,
        best_known: Dict[str, int],
        output_dir: Path | str = "./breakthrough_pursuit",
    ):
        """Initialize breakthrough pursuit system"""
        self.data_dir = Path(data_dir)
        self.best_known = best_known
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger()
        self.championship = QAPSquadChampionship(
            data_dir=data_dir, best_known=best_known, output_dir=self.output_dir
        )

        self.milestones: List[BreakthroughMilestone] = []
        self.load_milestone_history()

    def load_milestone_history(self):
        """Load previous milestone history"""
        history_file = self.output_dir / "milestone_history.json"
        if history_file.exists():
            with open(history_file) as f:
                data = json.load(f)
                # Reconstruct milestones
                for m in data.get("milestones", []):
                    milestone = BreakthroughMilestone(**m)
                    self.milestones.append(milestone)

    def save_milestone_history(self):
        """Save milestone history"""
        history_file = self.output_dir / "milestone_history.json"
        data = {"milestones": [asdict(m) for m in self.milestones]}
        with open(history_file, "w") as f:
            json.dump(data, f, indent=2)

    def run_weekly_cycle(
        self,
        tier_target: str = "entry",
        instances: Optional[List[str]] = None,
    ) -> BreakthroughMilestone:
        """Run one week of breakthrough pursuit"""
        week_num = len(self.milestones) + 1

        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"🎯 BREAKTHROUGH PURSUIT CYCLE - WEEK {week_num}")
        self.logger.info(f"Target Tier: {tier_target.upper()}")
        self.logger.info(f"{'='*70}\n")

        # Monday: Run championship
        self.logger.info("📊 MONDAY: Running championship...")
        results = self.championship.run_championship(instances=instances)
        self.championship.print_summary()

        # Tuesday: Analyze results
        self.logger.info("\n📈 TUESDAY: Analyzing for breakthroughs...")
        analysis = self.analyze_for_breakthrough(tier_target)

        # Wednesday: Identify opportunities
        self.logger.info("\n🔍 WEDNESDAY: Identifying opportunities...")
        opportunities = self.identify_optimization_opportunities(analysis)

        # Thursday: Design optimizations
        self.logger.info("\n💡 THURSDAY: Designing optimizations...")
        strategies = self.design_optimization_strategies(opportunities)

        # Friday: Create milestone report
        self.logger.info("\n📋 FRIDAY: Creating milestone report...")
        milestone = self.create_milestone_report(
            week_num, tier_target, analysis, opportunities, strategies
        )

        # Save
        self.milestones.append(milestone)
        self.save_milestone_history()

        # Report
        self.print_weekly_report(milestone)

        return milestone

    def analyze_for_breakthrough(self, tier_target: str) -> Dict:
        """Analyze championship results for breakthrough potential"""
        tier = self.TIERS[tier_target]

        # Extract stats
        stats_dict = self.championship.method_stats

        # Analyze each method
        analysis = {
            "tier_target": tier_target,
            "tier_def": asdict(tier),
            "methods_by_closeness": [],
            "breakthrough_candidates": [],
            "weak_areas": [],
            "opportunities": [],
        }

        for method_name, stats in sorted(stats_dict.items(), key=lambda x: x[1].avg_gap):
            # How close to breakthrough?
            gap_gap = stats.avg_gap - tier.gap_target
            consistency_gap = tier.consistency_target - stats.consistency_score
            win_rate_gap = tier.win_rate_target - stats.win_rate

            closeness_score = abs(gap_gap) + abs(consistency_gap) * 100 + abs(win_rate_gap) * 100

            method_info = {
                "method": method_name,
                "avg_gap": stats.avg_gap,
                "consistency": stats.consistency_score,
                "win_rate": stats.win_rate,
                "closeness_score": closeness_score,
                "gap_from_target": gap_gap,
                "consistency_from_target": consistency_gap,
                "win_rate_from_target": win_rate_gap,
            }

            analysis["methods_by_closeness"].append(method_info)

            # Is it a breakthrough candidate?
            gaps_met = stats.avg_gap <= tier.gap_target
            consistency_ok = stats.consistency_score >= tier.consistency_target * 0.9
            win_rate_ok = stats.win_rate >= tier.win_rate_target * 0.9

            if gaps_met and consistency_ok and win_rate_ok:
                analysis["breakthrough_candidates"].append(method_info)

        # Identify weak areas
        for result in self.championship.results:
            if result.gap_percent > tier.gap_target + 5:
                analysis["weak_areas"].append(
                    {
                        "instance": result.instance_name,
                        "method": result.method_name,
                        "gap": result.gap_percent,
                    }
                )

        return analysis

    def identify_optimization_opportunities(self, analysis: Dict) -> List[Dict]:
        """Identify specific optimization opportunities"""
        opportunities = []

        # Opportunity 1: Push closest method to breakthrough
        if analysis["methods_by_closeness"]:
            closest = analysis["methods_by_closeness"][0]
            opportunities.append(
                {
                    "type": "polish_top_method",
                    "method": closest["method"],
                    "gap_to_close": closest["gap_from_target"],
                    "action": f"Optimize {closest['method']}, need {abs(closest['gap_from_target']):.2f}% gap improvement",
                }
            )

        # Opportunity 2: Fix consistency issues
        low_consistency = [m for m in analysis["methods_by_closeness"] if m["consistency"] < 0.8]
        if low_consistency:
            opportunities.append(
                {
                    "type": "improve_consistency",
                    "methods": [m["method"] for m in low_consistency],
                    "action": "Focus on making methods more reliable (reduce variance)",
                }
            )

        # Opportunity 3: Tackle weak instances
        weak_instances = {}
        for weak in analysis["weak_areas"][:10]:
            inst = weak["instance"]
            if inst not in weak_instances:
                weak_instances[inst] = []
            weak_instances[inst].append(weak)

        if weak_instances:
            opportunities.append(
                {
                    "type": "target_weak_instances",
                    "instances": list(weak_instances.keys()),
                    "action": f"Develop specialized approaches for {len(weak_instances)} weak instances",
                }
            )

        # Opportunity 4: Hybrid combinations
        opportunities.append(
            {
                "type": "test_combinations",
                "action": "Test top 3-5 methods in different combinations/orderings",
            }
        )

        return opportunities

    def design_optimization_strategies(self, opportunities: List[Dict]) -> List[Dict]:
        """Design specific optimization strategies"""
        strategies = []

        for opp in opportunities:
            if opp["type"] == "polish_top_method":
                strategies.append(
                    {
                        "name": f"Polish {opp['method']}",
                        "steps": [
                            "1. Identify parameter sensitivity",
                            "2. Test parameter variations",
                            "3. Run 10+ seeds for consistency",
                            "4. Profile for bottlenecks",
                            f"5. Target: Close {opp['gap_to_close']:.2f}% gap",
                        ],
                    }
                )

            elif opp["type"] == "improve_consistency":
                strategies.append(
                    {
                        "name": "Consistency Improvement",
                        "steps": [
                            "1. Run each method 10x per instance",
                            "2. Analyze variance sources",
                            "3. Add noise-resistant techniques",
                            "4. Test with different seeds",
                            "5. Reduce std dev target: 50% reduction",
                        ],
                    }
                )

            elif opp["type"] == "target_weak_instances":
                strategies.append(
                    {
                        "name": "Weak Instance Specialization",
                        "steps": [
                            f"1. Analyze {len(opp['instances'])} weak instances",
                            "2. Identify common characteristics",
                            "3. Develop targeted method",
                            "4. Test vs baseline methods",
                            "5. Measure improvement",
                        ],
                    }
                )

            elif opp["type"] == "test_combinations":
                strategies.append(
                    {
                        "name": "Hybrid Method Combinations",
                        "steps": [
                            "1. Select top 3-5 methods",
                            "2. Test 5+ orderings",
                            "3. Test parallel combinations",
                            "4. Test adaptive switching",
                            "5. Measure synergy benefit",
                        ],
                    }
                )

        return strategies

    def create_milestone_report(
        self,
        week: int,
        tier_target: str,
        analysis: Dict,
        opportunities: List[Dict],
        strategies: List[Dict],
    ) -> BreakthroughMilestone:
        """Create weekly milestone report"""
        tier = self.TIERS[tier_target]

        # Current performance
        stats = self.championship.method_stats
        best_method = min(stats.values(), key=lambda x: x.avg_gap)

        breakthrough = (
            best_method.avg_gap <= tier.gap_target
            and best_method.consistency_score >= tier.consistency_target
            and best_method.win_rate >= tier.win_rate_target
        )

        # Get best method name
        best_method_name = min(stats.items(), key=lambda x: x[1].avg_gap)[0]

        milestone = BreakthroughMilestone(
            week=week,
            date=datetime.now().isoformat(),
            tier_target=tier_target,
            current_avg_gap=best_method.avg_gap,
            current_consistency=best_method.consistency_score,
            current_win_rate=best_method.win_rate,
            breakthrough_achieved=breakthrough,
            progress_notes=f"Best method: {best_method_name} | Gap: {best_method.avg_gap:.2f}% | Wins: {best_method.wins}",
            next_week_focus=[opp["action"] for opp in opportunities[:3]],
        )

        return milestone

    def print_weekly_report(self, milestone: BreakthroughMilestone):
        """Print beautiful weekly report"""
        report = "\n" + "=" * 70 + "\n"
        report += "📊 WEEKLY BREAKTHROUGH REPORT\n"
        report += "=" * 70 + "\n"

        report += f"Week: {milestone.week}\n"
        report += f"Target Tier: {milestone.tier_target.upper()}\n"
        report += f"Date: {milestone.date}\n\n"

        report += "PERFORMANCE:\n"
        report += f"  Current Gap: {milestone.current_avg_gap:.2f}%\n"
        report += f"  Consistency: {milestone.current_consistency:.1%}\n"
        report += f"  Win Rate: {milestone.current_win_rate:.1%}\n\n"

        tier = self.TIERS[milestone.tier_target]
        report += "BREAKTHROUGH TARGETS:\n"
        report += f"  Gap Target: {tier.gap_target:.2f}%\n"
        report += f"  Consistency Target: {tier.consistency_target:.1%}\n"
        report += f"  Win Rate Target: {tier.win_rate_target:.1%}\n\n"

        if milestone.breakthrough_achieved:
            report += "🏆 BREAKTHROUGH ACHIEVED! 🏆\n"
        else:
            gap_remaining = tier.gap_target - milestone.current_avg_gap
            report += "⏳ Breakthrough not yet achieved.\n"
            report += f"   Gap to close: {gap_remaining:.2f}%\n\n"

        report += "NEXT WEEK FOCUS:\n"
        for focus in milestone.next_week_focus:
            report += f"  • {focus}\n"

        report += "\n" + "=" * 70 + "\n"
        print(report)

    def continue_until_breakthrough(
        self,
        tier_target: str = "championship",
        max_weeks: int = 52,
        instances: Optional[List[str]] = None,
    ):
        """Run continuous cycles until breakthrough achieved"""
        self.logger.info("\n🚀 STARTING CONTINUOUS BREAKTHROUGH PURSUIT")
        self.logger.info(f"Target: {tier_target.upper()}")
        self.logger.info(f"Max weeks: {max_weeks}\n")

        for week in range(1, max_weeks + 1):
            milestone = self.run_weekly_cycle(tier_target, instances)

            if milestone.breakthrough_achieved:
                self.logger.info(f"\n🏆 BREAKTHROUGH ACHIEVED IN WEEK {week}! 🏆")
                self.print_celebration(milestone)
                return milestone

            # Calculate progress
            if len(self.milestones) > 1:
                prev = self.milestones[-2]
                improvement = prev.current_avg_gap - milestone.current_avg_gap
                self.logger.info(f"Week {week} improvement: {improvement:+.2f}%")

    def print_celebration(self, milestone: BreakthroughMilestone):
        """Print breakthrough celebration"""
        celebration = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                   🏆 BREAKTHROUGH ACHIEVED! 🏆                            ║
╚════════════════════════════════════════════════════════════════════════════╝

🎉 Congratulations! We reached the BREAKTHROUGH milestone!

BREAKTHROUGH DETAILS:
╔════════════════════════════════════════════════════════════════════════════╗
║ Tier: {milestone.tier_target.upper()}
║ Week: {milestone.week}
║ Date: {milestone.date}
║
║ Performance Achieved:
║  • Gap: {milestone.current_avg_gap:.2f}% (Target: <15%+)
║  • Consistency: {milestone.current_consistency:.1%}
║  • Win Rate: {milestone.current_win_rate:.1%}
║
║ This is a championship-quality result!
╚════════════════════════════════════════════════════════════════════════════╝

NEXT STEPS:
  1. Document the breakthrough
  2. Publish the results
  3. Deploy to production
  4. Continue optimizing for higher tiers

        """

        print(celebration)


__all__ = ["BreakthroughPursuitSystem", "BreakthroughTier", "BreakthroughMilestone"]
