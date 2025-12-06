"""
Plotting utilities for optimization visualization

Provides matplotlib-based plotting functions for analyzing optimization results.
Note: matplotlib is an optional dependency, install with: pip install Librex[viz]
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None  # type: ignore
    Figure = Any  # type: ignore
    Axes = Any  # type: ignore


def _check_matplotlib() -> None:
    """Check if matplotlib is available, raise helpful error if not"""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "Matplotlib is required for visualization. "
            "Install with: pip install Librex[viz] or pip install matplotlib"
        )


def plot_convergence(
    history: List[float],
    title: str = "Convergence Plot",
    xlabel: str = "Iteration",
    ylabel: str = "Objective Value",
    log_scale: bool = False,
    show_best: bool = True,
    ax: Optional[Axes] = None,
) -> Figure:
    """
    Plot convergence history

    Args:
        history: List of objective values over iterations
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        log_scale: Use logarithmic scale for y-axis
        show_best: Show horizontal line at best value
        ax: Optional matplotlib axes to plot on

    Returns:
        matplotlib Figure object

    Example:
        >>> from Librex import optimize
        >>> from Librex.visualization import plot_convergence
        >>>
        >>> result = optimize(problem, adapter, method='simulated_annealing')
        >>> fig = plot_convergence(result['history'])
        >>> plt.show()
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    iterations = np.arange(len(history))
    ax.plot(iterations, history, 'b-', linewidth=2, label='Objective')

    if show_best:
        best_value = min(history)
        ax.axhline(y=best_value, color='r', linestyle='--',
                   linewidth=1.5, label=f'Best: {best_value:.2f}')

    if log_scale:
        ax.set_yscale('log')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_method_comparison(
    results: List[Dict[str, Any]],
    metric: str = 'objective',
    title: str = "Method Comparison",
    ax: Optional[Axes] = None,
) -> Figure:
    """
    Create bar chart comparing multiple methods

    Args:
        results: List of result dicts with 'method' and metric keys
        metric: Metric to compare ('objective', 'runtime', etc.)
        title: Plot title
        ax: Optional matplotlib axes to plot on

    Returns:
        matplotlib Figure object

    Example:
        >>> from Librex.benchmarking import compare_methods
        >>> from Librex.visualization import plot_method_comparison
        >>>
        >>> results = compare_methods(problem, adapter,
        ...     methods=['random_search', 'simulated_annealing', 'genetic_algorithm'])
        >>>
        >>> # Convert BenchmarkResult to dict format
        >>> result_dicts = [
        ...     {'method': r.method, 'objective': r.objective_value,
        ...      'runtime': r.runtime_seconds}
        ...     for r in results
        ... ]
        >>>
        >>> fig = plot_method_comparison(result_dicts, metric='objective')
        >>> plt.show()
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    methods = [r.get('method', f"Method {i}") for i, r in enumerate(results)]
    values = [r.get(metric, 0) for r in results]

    # Create bar chart
    bars = ax.bar(methods, values, color='steelblue', alpha=0.7, edgecolor='black')

    # Highlight best (minimum for objective, could be max for other metrics)
    best_idx = np.argmin(values)
    bars[best_idx].set_color('forestgreen')
    bars[best_idx].set_alpha(0.9)

    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Rotate x labels if many methods
    if len(methods) > 5:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}',
                ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    return fig


def plot_performance_profile(
    profiles: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str = "Performance Profile",
    ax: Optional[Axes] = None,
) -> Figure:
    """
    Plot performance profiles for multiple methods

    Args:
        profiles: Dict mapping method name to (tau, probability) tuples
        title: Plot title
        ax: Optional matplotlib axes to plot on

    Returns:
        matplotlib Figure object

    Reference:
        Dolan & Moré (2002), "Benchmarking optimization software with
        performance profiles"

    Example:
        >>> from Librex.benchmarking.metrics import compute_performance_profile
        >>> from Librex.visualization import plot_performance_profile
        >>>
        >>> # Run benchmarks and collect results
        >>> method1_results = [...]  # Results on multiple problems
        >>> method2_results = [...]
        >>>
        >>> profiles = {
        ...     'Method 1': compute_performance_profile(method1_results),
        ...     'Method 2': compute_performance_profile(method2_results),
        ... }
        >>>
        >>> fig = plot_performance_profile(profiles)
        >>> plt.show()
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for i, (method, (tau, prob)) in enumerate(profiles.items()):
        color = colors[i % len(colors)]
        ax.plot(tau, prob, linewidth=2, label=method, color=color)

    ax.set_xlabel('Performance Ratio (τ)', fontsize=12)
    ax.set_ylabel('P(performance ratio ≤ τ)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=1.0)
    ax.set_ylim([0, 1])

    fig.tight_layout()
    return fig


def plot_solution_distribution(
    solutions: List[np.ndarray],
    objectives: List[float],
    title: str = "Solution Distribution",
    max_display: int = 50,
    ax: Optional[Axes] = None,
) -> Figure:
    """
    Visualize distribution of solutions and objectives

    Args:
        solutions: List of solution arrays
        objectives: List of corresponding objective values
        title: Plot title
        max_display: Maximum number of solutions to display
        ax: Optional matplotlib axes to plot on

    Returns:
        matplotlib Figure object

    Example:
        >>> # Collect solutions from multiple runs
        >>> solutions = []
        >>> objectives = []
        >>> for _ in range(20):
        ...     result = optimize(problem, adapter, method='genetic_algorithm')
        ...     solutions.append(result['solution'])
        ...     objectives.append(result['objective'])
        >>>
        >>> fig = plot_solution_distribution(solutions, objectives)
        >>> plt.show()
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()

    # Limit display if too many solutions
    if len(solutions) > max_display:
        indices = np.linspace(0, len(solutions) - 1, max_display, dtype=int)
        solutions = [solutions[i] for i in indices]
        objectives = [objectives[i] for i in indices]

    # Create scatter plot of objective values
    x_positions = np.arange(len(objectives))
    colors = plt.cm.viridis((np.array(objectives) - min(objectives)) /
                            (max(objectives) - min(objectives) + 1e-10))

    ax.scatter(x_positions, objectives, c=colors, s=100, alpha=0.6, edgecolors='black')

    # Add horizontal line at best
    best_obj = min(objectives)
    ax.axhline(y=best_obj, color='r', linestyle='--',
               linewidth=1.5, label=f'Best: {best_obj:.2f}')

    ax.set_xlabel('Solution Index', fontsize=12)
    ax.set_ylabel('Objective Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_benchmark_heatmap(
    suite_results: Dict[str, List[Any]],
    metric: str = 'objective',
    title: str = "Benchmark Heatmap",
    ax: Optional[Axes] = None,
) -> Figure:
    """
    Create heatmap of benchmark results across problems and methods

    Args:
        suite_results: Dict mapping problem name to list of BenchmarkResults
        metric: Metric to display ('objective', 'runtime', 'optimality_gap')
        title: Plot title
        ax: Optional matplotlib axes to plot on

    Returns:
        matplotlib Figure object

    Example:
        >>> from Librex.benchmarking import benchmark_suite
        >>> from Librex.visualization import plot_benchmark_heatmap
        >>>
        >>> problems = [
        ...     {'name': 'chr12a', 'instance': prob1, 'optimal': 9552},
        ...     {'name': 'chr15a', 'instance': prob2, 'optimal': 9896},
        ... ]
        >>>
        >>> suite_results = benchmark_suite(
        ...     problems, adapter,
        ...     methods=['simulated_annealing', 'genetic_algorithm']
        ... )
        >>>
        >>> fig = plot_benchmark_heatmap(suite_results, metric='optimality_gap')
        >>> plt.show()
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.get_figure()

    # Extract data
    problem_names = list(suite_results.keys())
    if not problem_names:
        raise ValueError("No benchmark results to plot")

    # Get method names from first problem
    method_names = [r.method for r in suite_results[problem_names[0]]]

    # Build data matrix
    data = np.zeros((len(problem_names), len(method_names)))

    for i, problem_name in enumerate(problem_names):
        for j, result in enumerate(suite_results[problem_name]):
            if metric == 'objective':
                value = result.objective_value
            elif metric == 'runtime':
                value = result.runtime_seconds
            elif metric == 'optimality_gap':
                value = result.optimality_gap if result.optimality_gap is not None else np.nan
            else:
                value = result.metadata.get(metric, np.nan)

            data[i, j] = value

    # Create heatmap
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(method_names)))
    ax.set_yticks(np.arange(len(problem_names)))
    ax.set_xticklabels(method_names)
    ax.set_yticklabels(problem_names)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(metric.replace('_', ' ').title(), rotation=-90, va="bottom")

    # Add text annotations
    for i in range(len(problem_names)):
        for j in range(len(method_names)):
            value = data[i, j]
            if not np.isnan(value):
                text = ax.text(j, i, f'{value:.2f}',
                             ha="center", va="center", color="black", fontsize=8)

    ax.set_title(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


__all__ = [
    'plot_convergence',
    'plot_method_comparison',
    'plot_performance_profile',
    'plot_solution_distribution',
    'plot_benchmark_heatmap',
]
