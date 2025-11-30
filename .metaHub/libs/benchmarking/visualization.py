"""
MetaHub Benchmarking Visualization

Chart generation and reporting for benchmark results.

Originally from: organizations/AlaweinOS/Benchmarks/visualize_benchmarks.py
Refactored: 2025-01-29
"""

from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    np = None
    plt = None


def create_performance_chart(
    data: Dict[str, Any],
    output_dir: Path,
    filename: str = "performance_chart.png"
) -> Optional[Path]:
    """
    Create performance comparison chart.

    Args:
        data: Benchmark results dictionary
        output_dir: Output directory
        filename: Output filename

    Returns:
        Path to created chart, or None if matplotlib unavailable
    """
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️  Matplotlib not available, skipping chart generation")
        return None

    results = data.get("results", [])
    if not results:
        print("No results to visualize")
        return None

    # Group by problem type
    problem_types = {}
    for r in results:
        if r.get('status') != 'success':
            continue

        ptype = r.get("problem_type", r.get("type", "default"))
        if ptype not in problem_types:
            problem_types[ptype] = {"sizes": [], "times": []}

        size = r.get("size", r.get("n", 0))
        problem_types[ptype]["sizes"].append(size)
        problem_types[ptype]["times"].append(r.get("mean", 0))

    if not problem_types:
        print("No data with problem types to visualize")
        return None

    # Create figure
    fig, axes = plt.subplots(1, len(problem_types), figsize=(15, 5))
    if len(problem_types) == 1:
        axes = [axes]

    for idx, (ptype, pdata) in enumerate(problem_types.items()):
        ax = axes[idx]

        sizes = pdata["sizes"]
        times = pdata["times"]

        ax.plot(sizes, times, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel("Problem Size", fontsize=12, fontweight='bold')
        ax.set_ylabel("Time (seconds)", fontsize=12, fontweight='bold')
        ax.set_title(f"{ptype} Performance", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = Path(output_dir) / filename
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Performance chart saved: {output_file}")
    plt.close()

    return output_file


def create_summary_table(
    data: Dict[str, Any],
    output_dir: Path,
    filename: str = "summary_table.png"
) -> Optional[Path]:
    """
    Create summary statistics table.

    Args:
        data: Benchmark results dictionary
        output_dir: Output directory
        filename: Output filename

    Returns:
        Path to created table, or None if matplotlib unavailable
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    summary = data.get("summary", {})
    if not summary or not summary.get('by_type'):
        return None

    # Create table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    # Table data
    table_data = []
    table_data.append(["Type", "Avg Time (s)", "Std (s)", "Min (s)", "Max (s)"])

    for ptype, stats in summary.get("by_type", {}).items():
        table_data.append([
            ptype,
            f"{stats['mean']:.6f}",
            f"{stats['stdev']:.6f}",
            f"{stats['min']:.6f}",
            f"{stats['max']:.6f}",
        ])

    table = ax.table(
        cellText=table_data,
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.15, 0.15, 0.15, 0.15]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    plt.title("Benchmark Summary", fontsize=16, fontweight='bold', pad=20)

    output_file = Path(output_dir) / filename
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Summary table saved: {output_file}")
    plt.close()

    return output_file


def generate_markdown_report(
    data: Dict[str, Any],
    output_dir: Path,
    filename: str = "BENCHMARK_REPORT.md"
) -> Path:
    """
    Generate markdown benchmark report.

    Args:
        data: Benchmark results dictionary
        output_dir: Output directory
        filename: Output filename

    Returns:
        Path to created report
    """
    report = []
    report.append("# Benchmark Report\n")
    report.append(f"**Timestamp:** {data.get('timestamp', 'Unknown')}\n")
    report.append("\n---\n")

    summary = data.get("summary", {})

    report.append("\n## Summary Statistics\n")
    report.append(f"- **Total Benchmarks:** {summary.get('total_benchmarks', 0)}\n")
    report.append(f"- **Successful:** {summary.get('successful', 0)}\n")
    report.append(f"- **Failed:** {summary.get('failed', 0)}\n")

    if summary.get('by_type'):
        report.append("\n### Performance by Type\n")
        report.append("\n| Type | Avg Time (s) | Std Dev | Min | Max |\n")
        report.append("|------|--------------|---------|-----|-----|\n")

        for ptype, stats in summary.get("by_type", {}).items():
            report.append(
                f"| {ptype} | {stats['mean']:.6f} | {stats['stdev']:.6f} | "
                f"{stats['min']:.6f} | {stats['max']:.6f} |\n"
            )

    report.append("\n---\n")
    report.append("\n*Generated by MetaHub Benchmarking System*\n")

    output_file = Path(output_dir) / filename
    with open(output_file, "w") as f:
        f.writelines(report)

    print(f"✅ Markdown report saved: {output_file}")
    return output_file
