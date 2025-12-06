"""
QAP Squad Championship - Advanced Visualization System

Creates publication-quality visualizations for:
- Gap heatmaps (methods × instances)
- Time vs Quality efficiency frontier
- Win rate comparisons
- Performance trajectories by problem size
- Method synergy analysis
"""

from pathlib import Path
from typing import Dict, List, Optional

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np

from .benchmarking_suite import BenchmarkEntry
from .plots_base import setup_publication_style


class ChampionshipVisualizer:
    """Advanced visualization for QAP Squad Championship"""

    def __init__(self, results: List[BenchmarkEntry], output_dir: Path | str = "./visualizations"):
        """
        Initialize visualizer.

        Args:
            results: List of benchmark results
            output_dir: Directory for saving visualizations
        """
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        setup_publication_style()

    def create_gap_heatmap(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create heatmap of gaps: methods × instances.

        Green (low gap) to Red (high gap)
        """
        # Build matrix: rows=methods, cols=instances
        methods = sorted(set(e.method_name for e in self.results))
        instances = sorted(set(e.instance_name for e in self.results))

        gap_matrix = np.full((len(methods), len(instances)), np.nan)

        for i, method in enumerate(methods):
            for j, instance in enumerate(instances):
                entries = [
                    e
                    for e in self.results
                    if e.method_name == method and e.instance_name == instance
                ]
                if entries:
                    gap_matrix[i, j] = entries[0].gap_percent

        fig, ax = plt.subplots(figsize=(14, 8))

        # Custom colormap: green (low) → yellow → red (high)
        cmap = LinearSegmentedColormap.from_list("gap_cmap", ["#1B5E20", "#FFF176", "#C62828"])

        im = ax.imshow(
            gap_matrix, cmap=cmap, aspect="auto", vmin=0, vmax=np.nanpercentile(gap_matrix, 95)
        )

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Optimality Gap (%)", rotation=270, labelpad=20)

        # Axes
        ax.set_xticks(np.arange(len(instances)))
        ax.set_yticks(np.arange(len(methods)))
        ax.set_xticklabels(instances, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(methods, fontsize=9)

        ax.set_xlabel("Instance", fontsize=12)
        ax.set_ylabel("Method", fontsize=12)
        ax.set_title(
            "Gap Heatmap: Methods × Instances (Lower is Better)", fontsize=14, fontweight="bold"
        )

        # Add gap values as text
        for i in range(len(methods)):
            for j in range(len(instances)):
                if not np.isnan(gap_matrix[i, j]):
                    val = gap_matrix[i, j]
                    text_color = "white" if val > 15 else "black"
                    ax.text(
                        j, i, f"{val:.1f}", ha="center", va="center", color=text_color, fontsize=7
                    )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def create_efficiency_frontier(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create Time vs Quality plot showing efficiency frontier.
        Points on frontier are Pareto-optimal.
        """
        # Aggregate by method: avg gap vs avg time
        method_data = {}

        for result in self.results:
            if result.gap_percent < float("inf"):
                if result.method_name not in method_data:
                    method_data[result.method_name] = {"gaps": [], "times": []}

                method_data[result.method_name]["gaps"].append(result.gap_percent)
                method_data[result.method_name]["times"].append(result.time_seconds)

        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot each method
        colors = plt.cm.tab20(np.linspace(0, 1, len(method_data)))

        for (method_name, data), color in zip(method_data.items(), colors):
            avg_gap = np.mean(data["gaps"])
            avg_time = np.mean(data["times"])

            ax.scatter(
                avg_time,
                avg_gap,
                s=200,
                alpha=0.7,
                color=color,
                edgecolors="black",
                linewidth=1.5,
                label=method_name,
            )

        # Compute and highlight Pareto frontier
        methods_sorted = sorted(
            method_data.items(), key=lambda x: (np.mean(x[1]["gaps"]), np.mean(x[1]["times"]))
        )

        frontier = []
        for method_name, data in methods_sorted:
            avg_gap = np.mean(data["gaps"])
            avg_time = np.mean(data["times"])

            # Point is on frontier if no other point is better on both dimensions
            is_frontier = True
            for other_name, other_data in methods_sorted:
                if method_name != other_name:
                    other_gap = np.mean(other_data["gaps"])
                    other_time = np.mean(other_data["times"])
                    if other_gap <= avg_gap and other_time <= avg_time:
                        is_frontier = False
                        break

            if is_frontier:
                frontier.append((avg_time, avg_gap, method_name))

        # Draw frontier
        if frontier:
            frontier_sorted = sorted(frontier, key=lambda x: x[0])
            frontier_times = [x[0] for x in frontier_sorted]
            frontier_gaps = [x[1] for x in frontier_sorted]
            ax.plot(
                frontier_times,
                frontier_gaps,
                "g--",
                linewidth=2,
                alpha=0.5,
                label="Pareto Frontier",
            )

        ax.set_xlabel("Average Time (seconds)", fontsize=12)
        ax.set_ylabel("Average Gap (%)", fontsize=12)
        ax.set_title(
            "Efficiency Frontier: Quality vs Time\n(Lower and Left is Better)",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def create_win_rate_chart(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create bar chart of win rates by method"""
        # Count wins for each method
        method_wins = {}

        for instance_name in set(e.instance_name for e in self.results):
            instance_results = [
                e
                for e in self.results
                if e.instance_name == instance_name and e.gap_percent < float("inf")
            ]

            if instance_results:
                best_gap = min(e.gap_percent for e in instance_results)
                winners = [e.method_name for e in instance_results if e.gap_percent == best_gap]

                for method in winners:
                    method_wins[method] = method_wins.get(method, 0) + 1 / len(winners)

        total_instances = len(set(e.instance_name for e in self.results))

        # Sort by wins
        sorted_methods = sorted(method_wins.items(), key=lambda x: -x[1])

        fig, ax = plt.subplots(figsize=(12, 6))

        methods = [m[0] for m in sorted_methods]
        wins = [m[1] for m in sorted_methods]
        win_rates = [w / total_instances * 100 for w in wins]

        bars = ax.barh(
            range(len(methods)), win_rates, color="steelblue", alpha=0.8, edgecolor="black"
        )

        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods, fontsize=10)
        ax.set_xlabel("Win Rate (%)", fontsize=12)
        ax.set_title(
            "Method Win Rates (% of instances where method was best)",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, axis="x")

        # Annotate with percentages
        for i, (bar, rate) in enumerate(zip(bars, win_rates)):
            ax.text(rate + 1, i, f"{rate:.1f}%", va="center", fontsize=9, fontweight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def create_performance_by_size(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create performance trajectory by problem size.
        Shows how gap changes as problems get harder.
        """
        # Categorize instances by size
        size_categories = {"micro": [], "small": [], "medium": [], "large": []}

        for instance_name in set(e.instance_name for e in self.results):
            import re

            match = re.search(r"(\d+)", instance_name)
            if match:
                n = int(match.group(1))
                if n < 20:
                    size_categories["micro"].append(instance_name)
                elif n < 40:
                    size_categories["small"].append(instance_name)
                elif n < 60:
                    size_categories["medium"].append(instance_name)
                else:
                    size_categories["large"].append(instance_name)

        fig, ax = plt.subplots(figsize=(12, 6))

        methods = sorted(set(e.method_name for e in self.results))
        size_order = ["micro", "small", "medium", "large"]

        for method in methods[:10]:  # Top 10 methods for clarity
            gaps_by_size = []

            for size_cat in size_order:
                instance_results = [
                    e
                    for e in self.results
                    if e.method_name == method
                    and e.instance_name in size_categories[size_cat]
                    and e.gap_percent < float("inf")
                ]

                if instance_results:
                    avg_gap = np.mean([e.gap_percent for e in instance_results])
                    gaps_by_size.append(avg_gap)
                else:
                    gaps_by_size.append(None)

            ax.plot(range(len(size_order)), gaps_by_size, marker="o", label=method, linewidth=2)

        ax.set_xticks(range(len(size_order)))
        ax.set_xticklabels(size_order)
        ax.set_xlabel("Problem Size", fontsize=12)
        ax.set_ylabel("Average Gap (%)", fontsize=12)
        ax.set_title(
            "Performance Trajectory by Problem Size\n(Lower is Better)",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def create_consistency_analysis(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create scatter plot: Average Gap vs Consistency Score.
        Ideal methods are low gap + high consistency (bottom right).
        """
        method_stats_dict = {}

        for method_name in set(e.method_name for e in self.results):
            gaps = [
                e.gap_percent
                for e in self.results
                if e.method_name == method_name and e.gap_percent < float("inf")
            ]

            if gaps:
                avg_gap = np.mean(gaps)
                std_gap = np.std(gaps) if len(gaps) > 1 else 0
                consistency = 1 - (std_gap / max(avg_gap, 1e-6))

                method_stats_dict[method_name] = {
                    "avg_gap": avg_gap,
                    "consistency": consistency,
                    "num_runs": len(gaps),
                }

        fig, ax = plt.subplots(figsize=(12, 7))

        for method_name, stats in method_stats_dict.items():
            ax.scatter(
                stats["avg_gap"],
                stats["consistency"],
                s=300,
                alpha=0.7,
                edgecolors="black",
                linewidth=1.5,
            )
            ax.annotate(
                method_name, (stats["avg_gap"], stats["consistency"]), fontsize=8, ha="right"
            )

        ax.set_xlabel("Average Gap (%)", fontsize=12)
        ax.set_ylabel("Consistency Score (1.0 = Perfect)", fontsize=12)
        ax.set_title(
            "Method Consistency vs Quality\n(Bottom-Right is Ideal)", fontsize=14, fontweight="bold"
        )
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.1)

        # Highlight ideal region
        ax.axhspan(0.8, 1.1, alpha=0.1, color="green", label="Ideal Region")
        ax.axvspan(0, 10, alpha=0.1, color="blue")
        ax.legend(fontsize=10)

        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def generate_all_visualizations(self) -> Dict[str, Path]:
        """Generate all visualizations and save"""
        visualizations = {}

        print("\n📊 Generating Championship Visualizations...")

        print("  • Gap Heatmap...")
        heatmap_path = self.output_dir / "01_gap_heatmap.png"
        self.create_gap_heatmap(save_path=str(heatmap_path))
        visualizations["heatmap"] = heatmap_path

        print("  • Efficiency Frontier...")
        frontier_path = self.output_dir / "02_efficiency_frontier.png"
        self.create_efficiency_frontier(save_path=str(frontier_path))
        visualizations["frontier"] = frontier_path

        print("  • Win Rate Chart...")
        winrate_path = self.output_dir / "03_win_rates.png"
        self.create_win_rate_chart(save_path=str(winrate_path))
        visualizations["winrate"] = winrate_path

        print("  • Performance by Size...")
        size_path = self.output_dir / "04_performance_by_size.png"
        self.create_performance_by_size(save_path=str(size_path))
        visualizations["size"] = size_path

        print("  • Consistency Analysis...")
        consistency_path = self.output_dir / "05_consistency.png"
        self.create_consistency_analysis(save_path=str(consistency_path))
        visualizations["consistency"] = consistency_path

        print(f"\n✅ Visualizations saved to: {self.output_dir}\n")

        return visualizations


__all__ = ["ChampionshipVisualizer"]
