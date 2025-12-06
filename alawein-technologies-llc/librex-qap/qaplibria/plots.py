"""
Librex.QAP Plotting Module
==========================

Publication-quality visualizations for QAP benchmarks.

Key plots:
- 3-panel convergence (objective, DS violation, gradient norm)
- Gap heatmaps (methods × instances)
- Permutation comparisons (initial vs final)
- Eigenvalue analysis (Jacobian spectrum)
- Librex search (basin visualization)
- DS residual tracking

All plots use publication-ready styling with:
- Colorblind-safe palettes
- High DPI (300)
- Professional fonts
- Clear labels and legends

Author: Meshal Alawein
Date: 2025-10-16
"""

from typing import Dict, List, Optional

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from .plots_base import (
    COLORBLIND_PALETTE,
    add_gate_line,
    configure_single_axis,
    create_figure_with_subplots,
    save_and_show,
)

# Infrastructure imports
from .validation import validate_history_dict


def load_style(style_path: Optional[str] = None):
    """
    Load matplotlib style settings.

    Parameters
    ----------
    style_path : str, optional
        Path to style.yaml file
        If None, uses default publication style
    """
    if style_path:
        # Load from YAML (requires PyYAML)
        import yaml

        with open(style_path) as f:
            style_dict = yaml.safe_load(f)
        plt.rcParams.update(style_dict)
    else:
        # Default publication style
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["Times New Roman"],
                "font.size": 11,
                "axes.labelsize": 12,
                "axes.titlesize": 13,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
                "figure.dpi": 150,
                "savefig.dpi": 300,
                "savefig.format": "pdf",
                "savefig.bbox": "tight",
                "axes.grid": True,
                "grid.alpha": 0.3,
                "grid.linestyle": "--",
                "lines.linewidth": 2,
                "axes.linewidth": 1.2,
            }
        )


# Color schemes
COLORS = {
    "novel": "#2E7D32",  # Dark green for novel methods
    "baseline": "#C62828",  # Dark red for baseline methods
    "neutral": "#1976D2",  # Blue for neutral/combined
    "accent": "#F57C00",  # Orange for highlights
}


def plot_convergence_3panel(
    history: Dict,
    title: str = "Convergence Analysis",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Create 3-panel convergence plot.

    Panels:
    1. Objective value vs time
    2. DS violation vs time (log scale)
    3. Gradient norm vs time (log scale)

    Parameters
    ----------
    history : dict
        History dictionary with keys:
        - 'times': Time points
        - 'objectives': Objective values
        - 'ds_violations': Constraint violations
        - 'grad_norms': Gradient norms
    title : str
        Main title for figure
    save_path : str, optional
        Path to save figure (PDF)
    show : bool
        Whether to display figure

    Returns
    -------
    fig : matplotlib.Figure
        Figure object
    """
    # Input validation
    validate_history_dict(history)

    fig, axes = create_figure_with_subplots(1, 3, figsize=(15, 4))
    times = history["times"]

    # Panel 1: Objective
    axes[0].plot(times, history["objectives"], color=COLORS["novel"], linewidth=2)
    configure_single_axis(
        axes[0], xlabel="Time (s)", ylabel="Objective Value", title="QAP Objective"
    )

    # Panel 2: DS Violation
    axes[1].plot(times, history["ds_violations"], color=COLORS["baseline"], linewidth=2)
    configure_single_axis(
        axes[1],
        xlabel="Time (s)",
        ylabel="DS Violation",
        title="Doubly-Stochastic Constraint",
        yscale="log",
    )
    add_gate_line(axes[1], 0.01, label="Gate (0.01)", linestyle="--")
    axes[1].legend()

    # Panel 3: Gradient Norm
    axes[2].plot(times, history["grad_norms"], color=COLORS["neutral"], linewidth=2)
    configure_single_axis(
        axes[2], xlabel="Time (s)", ylabel="Gradient Norm", title="Gradient Magnitude", yscale="log"
    )

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    save_and_show(fig, save_path, show)

    return fig


def plot_gap_heatmap(
    results_df,
    method_col: str = "method",
    instance_col: str = "instance",
    gap_col: str = "gap",
    title: str = "Optimality Gap Heatmap",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Create heatmap of optimality gaps (methods × instances).

    Parameters
    ----------
    results_df : pandas.DataFrame
        Benchmark results with columns for method, instance, gap
    method_col, instance_col, gap_col : str
        Column names
    title : str
        Figure title
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display

    Returns
    -------
    fig : matplotlib.Figure
    """

    # Pivot to matrix form
    gap_matrix = results_df.pivot(index=method_col, columns=instance_col, values=gap_col)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Custom colormap: green (low gap) to red (high gap)
    cmap = LinearSegmentedColormap.from_list("gap_cmap", ["#1B5E20", "#FFF176", "#C62828"])

    im = ax.imshow(
        gap_matrix.values, cmap=cmap, aspect="auto", vmin=0, vmax=min(50, gap_matrix.max().max())
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Optimality Gap (%)", rotation=270, labelpad=20)

    # Axes
    ax.set_xticks(np.arange(len(gap_matrix.columns)))
    ax.set_yticks(np.arange(len(gap_matrix.index)))
    ax.set_xticklabels(gap_matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(gap_matrix.index)

    ax.set_xlabel("Instance")
    ax.set_ylabel("Method")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    # Add gap values as text
    for i in range(len(gap_matrix.index)):
        for j in range(len(gap_matrix.columns)):
            value = gap_matrix.values[i, j]
            if not np.isnan(value):
                text_color = "white" if value > 25 else "black"
                ax.text(
                    j, i, f"{value:.1f}", ha="center", va="center", color=text_color, fontsize=9
                )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_permutation_comparison(
    X_init: np.ndarray,
    X_final: np.ndarray,
    P_final: Optional[np.ndarray] = None,
    title: str = "Solution Evolution",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Visualize initial DS matrix, final DS matrix, and final permutation.

    Parameters
    ----------
    X_init : np.ndarray
        Initial doubly-stochastic matrix (n x n)
    X_final : np.ndarray
        Final doubly-stochastic matrix (n x n)
    P_final : np.ndarray, optional
        Final permutation matrix (n x n)
        If None, only shows 2 panels
    title : str
        Figure title
    save_path : str, optional
        Path to save
    show : bool
        Whether to display

    Returns
    -------
    fig : matplotlib.Figure
    """
    n_panels = 3 if P_final is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))

    if n_panels == 2:
        axes = [axes[0], axes[1]]
    else:
        axes = [axes[0], axes[1], axes[2]]

    # Panel 1: Initial
    im1 = axes[0].imshow(X_init, cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title("Initial (Spectral)")
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    # Panel 2: Final DS
    im2 = axes[1].imshow(X_final, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("Final (Continuous)")
    axes[1].set_xlabel("Column")
    axes[1].set_ylabel("Row")
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    # Panel 3: Final Permutation (if provided)
    if P_final is not None:
        im3 = axes[2].imshow(P_final, cmap="binary", vmin=0, vmax=1)
        axes[2].set_title("Final (Rounded)")
        axes[2].set_xlabel("Column")
        axes[2].set_ylabel("Row")
        plt.colorbar(im3, ax=axes[2], fraction=0.046)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_eigenvalue_analysis(
    jacobian: np.ndarray,
    title: str = "Jacobian Eigenvalue Spectrum",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Analyze Jacobian eigenvalues for saddle detection.

    Shows:
    - Real vs imaginary parts
    - Histogram of real parts
    - Stability classification

    Parameters
    ----------
    jacobian : np.ndarray
        Jacobian matrix at equilibrium point
    title : str
        Figure title
    save_path : str, optional
        Path to save
    show : bool
        Whether to display

    Returns
    -------
    fig : matplotlib.Figure
    """
    # Compute eigenvalues
    eigvals = np.linalg.eigvals(jacobian)
    real_parts = np.real(eigvals)
    imag_parts = np.imag(eigvals)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Complex plane
    axes[0].scatter(
        real_parts,
        imag_parts,
        alpha=0.6,
        s=50,
        c=COLORBLIND_PALETTE[0],
        edgecolors="black",
        linewidths=0.5,
    )
    axes[0].axvline(x=0, color="red", linestyle="--", linewidth=1.5, label="Stability Boundary")
    axes[0].axhline(y=0, color="gray", linestyle=":", linewidth=1)
    axes[0].set_xlabel("Real Part")
    axes[0].set_ylabel("Imaginary Part")
    axes[0].set_title("Eigenvalue Spectrum")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Histogram of real parts
    axes[1].hist(real_parts, bins=30, color=COLORBLIND_PALETTE[1], alpha=0.7, edgecolor="black")
    axes[1].axvline(x=0, color="red", linestyle="--", linewidth=2, label="Stability Boundary")
    axes[1].set_xlabel("Real Part of Eigenvalue")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Real Part Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    # Stability classification
    n_positive = np.sum(real_parts > 1e-6)
    n_negative = np.sum(real_parts < -1e-6)
    n_zero = np.sum(np.abs(real_parts) <= 1e-6)

    classification = f"Positive: {n_positive}, Negative: {n_negative}, Near-zero: {n_zero}\n"
    if n_positive > 0 and n_negative > 0:
        classification += "Type: SADDLE POINT"
    elif n_positive == 0:
        classification += "Type: STABLE (Attractor)"
    else:
        classification += "Type: UNSTABLE (Repeller)"

    fig.text(
        0.5,
        -0.05,
        classification,
        ha="center",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_Librex_search(
    Librex_data: List[Dict],
    title: str = "Librex Search Results",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Visualize Librex found during basin search.

    Parameters
    ----------
    Librex_data : list of dict
        Each dict contains:
        - 'objective': Objective value at equilibrium
        - 'type': 'attractor', 'saddle', or 'repeller'
        - 'basin_size': Estimate of basin volume (optional)
    title : str
        Figure title
    save_path : str, optional
        Path to save
    show : bool
        Whether to display

    Returns
    -------
    fig : matplotlib.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Extract data
    objectives = [eq["objective"] for eq in Librex_data]
    types = [eq["type"] for eq in Librex_data]
    basin_sizes = [eq.get("basin_size", 1.0) for eq in Librex_data]

    # Color mapping
    type_colors = {
        "attractor": COLORS["novel"],
        "saddle": COLORS["accent"],
        "repeller": COLORS["baseline"],
    }
    colors = [type_colors[t] for t in types]

    # Panel 1: Scatter plot (objective vs basin size)
    axes[0].scatter(
        objectives, basin_sizes, c=colors, s=100, alpha=0.7, edgecolors="black", linewidths=1
    )
    axes[0].set_xlabel("Objective Value")
    axes[0].set_ylabel("Basin Size (relative)")
    axes[0].set_title("Librex Landscape")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)

    # Legend
    legend_elements = [
        mpatches.Patch(color=COLORS["novel"], label="Attractor"),
        mpatches.Patch(color=COLORS["accent"], label="Saddle"),
        mpatches.Patch(color=COLORS["baseline"], label="Repeller"),
    ]
    axes[0].legend(handles=legend_elements, loc="best")

    # Panel 2: Type distribution
    type_counts = {t: types.count(t) for t in ["attractor", "saddle", "repeller"]}
    type_labels = list(type_counts.keys())
    type_values = list(type_counts.values())
    type_colors_list = [type_colors[t] for t in type_labels]

    axes[1].bar(
        type_labels,
        type_values,
        color=type_colors_list,
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
    )
    axes[1].set_xlabel("Equilibrium Type")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Type Distribution")
    axes[1].grid(True, alpha=0.3, axis="y")

    # Add counts on bars
    for i, (label, value) in enumerate(zip(type_labels, type_values)):
        axes[1].text(i, value + 0.5, str(value), ha="center", va="bottom", fontweight="bold")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_ds_residual_tracking(
    history: Dict,
    gates: Optional[Dict[str, float]] = None,
    title: str = "DS Residual Evolution",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Track doubly-stochastic constraint residuals over time.

    Shows row and column violations separately if available.

    Parameters
    ----------
    history : dict
        History with 'times' and 'ds_violations'
        Optionally 'row_violations' and 'col_violations'
    gates : dict, optional
        Performance gates, e.g., {'ds_violation': 0.01}
    title : str
        Figure title
    save_path : str, optional
        Path to save
    show : bool
        Whether to display

    Returns
    -------
    fig : matplotlib.Figure
    """
    # Input validation
    validate_history_dict(history)

    fig, ax = plt.subplots(figsize=(10, 6))
    times = history["times"]

    # Plot combined DS violation
    ax.plot(
        times, history["ds_violations"], color=COLORS["novel"], linewidth=2.5, label="DS Violation"
    )

    # Plot separate row/col violations if available
    if "row_violations" in history:
        ax.plot(
            times,
            history["row_violations"],
            color=COLORBLIND_PALETTE[1],
            linewidth=1.5,
            linestyle="--",
            alpha=0.7,
            label="Row Violation",
        )

    if "col_violations" in history:
        ax.plot(
            times,
            history["col_violations"],
            color=COLORBLIND_PALETTE[2],
            linewidth=1.5,
            linestyle="-.",
            alpha=0.7,
            label="Col Violation",
        )

    # Gate line
    if gates and "ds_violation" in gates:
        gate_value = gates["ds_violation"]
        add_gate_line(ax, gate_value, label=f"Gate ({gate_value})", linestyle="--")

    configure_single_axis(
        ax, xlabel="Time (s)", ylabel="Constraint Violation", title=title, yscale="log"
    )
    ax.legend(loc="best")

    save_and_show(fig, save_path, show)
    return fig


def plot_method_comparison_bars(
    results_df,
    metric: str = "gap",
    novel_methods: Optional[List[str]] = None,
    baseline_methods: Optional[List[str]] = None,
    title: str = "Method Comparison",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Bar chart comparing methods on a metric (e.g., gap or time).

    Color-codes novel vs baseline methods.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Results with 'method' and metric columns
    metric : str
        Column name to plot (e.g., 'gap', 'time')
    novel_methods : list of str, optional
        List of novel method names
    baseline_methods : list of str, optional
        List of baseline method names
    title : str
        Figure title
    save_path : str, optional
        Path to save
    show : bool
        Whether to display

    Returns
    -------
    fig : matplotlib.Figure
    """

    # Aggregate by method
    method_avg = results_df.groupby("method")[metric].mean().sort_values()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Determine colors
    colors_list = []
    for method in method_avg.index:
        if novel_methods and method in novel_methods:
            colors_list.append(COLORS["novel"])
        elif baseline_methods and method in baseline_methods:
            colors_list.append(COLORS["baseline"])
        else:
            colors_list.append(COLORS["neutral"])

    # Plot bars
    bars = ax.barh(
        range(len(method_avg)),
        method_avg.values,
        color=colors_list,
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )

    ax.set_yticks(range(len(method_avg)))
    ax.set_yticklabels(method_avg.index)
    ax.set_xlabel(metric.capitalize())
    ax.set_ylabel("Method")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for i, value in enumerate(method_avg.values):
        ax.text(value + 0.5, i, f"{value:.2f}", va="center", fontsize=9)

    # Legend
    if novel_methods or baseline_methods:
        legend_elements = []
        if novel_methods:
            legend_elements.append(mpatches.Patch(color=COLORS["novel"], label="Novel"))
        if baseline_methods:
            legend_elements.append(mpatches.Patch(color=COLORS["baseline"], label="Baseline"))
        ax.legend(handles=legend_elements, loc="best")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


# ============================================================================
# ADDITIONAL VISUALIZATIONS (from VISUALIZATION_SPECS.md)
# ============================================================================


def plot_objective_annotated(
    history: Dict,
    title: str = "Objective Evolution with Events",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot objective with event annotations (saddle detection, escapes, local search).

    Parameters
    ----------
    history : dict
        Keys: 'time', 'objective', 'saddle_detections', 'escapes', 'local_search_start'
    title : str
        Figure title
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display

    Returns
    -------
    fig : matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Main objective curve
    ax.plot(
        history["times"],
        history["objectives"],
        color=COLORS["novel"],
        linewidth=2.5,
        label="Objective",
    )

    # Mark saddle detections
    if "saddle_detections" in history and history["saddle_detections"]:
        for t in history["saddle_detections"]:
            ax.axvline(x=t, color=COLORS["accent"], linestyle=":", alpha=0.7, linewidth=1.5)
        ax.plot([], [], color=COLORS["accent"], linestyle=":", label="Saddle Detected")

    # Mark escapes
    if "escapes" in history and history["escapes"]:
        for t in history["escapes"]:
            ax.axvline(x=t, color="red", linestyle="--", alpha=0.7, linewidth=1.5)
        ax.plot([], [], color="red", linestyle="--", label="Reverse-Time Escape")

    # Mark local search
    if "local_search_start" in history:
        t_ls = history["local_search_start"]
        ax.axvline(x=t_ls, color="green", linestyle="-", alpha=0.7, linewidth=2)
        ax.text(t_ls, ax.get_ylim()[1] * 0.9, "Local Search", rotation=90, va="top", fontsize=10)

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Objective f(X)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_evolution_sequence(
    X_sequence: List[np.ndarray],
    times: List[float],
    title: str = "Solution Evolution (4-Frame)",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot 4-frame evolution of doubly-stochastic matrix.

    Shows X at t=0, T/4, T/2, T (or provided times).

    Parameters
    ----------
    X_sequence : list of np.ndarray
        Sequence of X matrices [X_0, X_{T/4}, X_{T/2}, X_T]
    times : list of float
        Corresponding timestamps
    title : str
        Figure title
    save_path : str, optional
        Path to save
    show : bool
        Whether to display

    Returns
    -------
    fig : matplotlib.Figure
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for i, (X, t, ax) in enumerate(zip(X_sequence, times, axes)):
        im = ax.imshow(X, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"t = {t:.2f}s", fontsize=12, fontweight="bold")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_energy_histogram(
    objectives: List[float],
    best_known: float,
    instance_name: str = "Had12",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot histogram of objectives from multi-start optimization.

    Parameters
    ----------
    objectives : list of float
        Final objectives from different starts
    best_known : float
        Known optimal value
    instance_name : str
        Instance name for labeling
    title : str, optional
        Figure title
    save_path : str, optional
        Path to save
    show : bool
        Whether to display

    Returns
    -------
    fig : matplotlib.Figure
    """
    if title is None:
        title = f"Multi-Start Energy Distribution ({instance_name})"

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    ax.hist(objectives, bins=15, color=COLORS["neutral"], alpha=0.7, edgecolor="black")

    # Mark best known
    ax.axvline(
        x=best_known,
        color="red",
        linestyle="--",
        linewidth=2.5,
        label=f"Best Known: {best_known:,.0f}",
    )

    # Mark found minimum
    found_min = min(objectives)
    ax.axvline(
        x=found_min,
        color="green",
        linestyle="-",
        linewidth=2.5,
        label=f"Found Best: {found_min:,.0f}",
    )

    ax.set_xlabel("Objective Value", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Print statistics
    unique = len(set([round(o, -2) for o in objectives]))
    gap_pct = (found_min - best_known) / best_known * 100

    stats_text = f"N={len(objectives)} starts | Unique levels={unique} | Gap={gap_pct:.2f}%"
    ax.text(
        0.5,
        -0.12,
        stats_text,
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_basin_clustering(
    X_finals: List[np.ndarray],
    objectives: Optional[List[float]] = None,
    title: str = "Basin Clustering Analysis",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot basin clustering via PCA projection and k-means.

    Parameters
    ----------
    X_finals : list of np.ndarray
        Final matrices from different optimization starts
    objectives : list of float, optional
        Objective values for coloring
    title : str
        Figure title
    save_path : str, optional
        Path to save
    show : bool
        Whether to display

    Returns
    -------
    fig : matplotlib.Figure
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    # Flatten matrices
    X_flat = np.array([X.flatten() for X in X_finals])

    # PCA to 2D
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_flat)

    # K-means clustering
    n_clusters = min(5, max(2, len(X_finals) // 3))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_2d)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Cluster visualization
    scatter = axes[0].scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=labels,
        cmap="tab10",
        s=150,
        alpha=0.7,
        edgecolors="black",
        linewidth=1.5,
    )

    # Mark cluster centers
    centers_2d = kmeans.cluster_centers_
    axes[0].scatter(
        centers_2d[:, 0],
        centers_2d[:, 1],
        c="red",
        marker="X",
        s=400,
        edgecolors="black",
        linewidth=2,
        label="Cluster Centers",
    )

    axes[0].set_xlabel("PC1", fontsize=12)
    axes[0].set_ylabel("PC2", fontsize=12)
    axes[0].set_title("Basin Clustering (PCA Projection)", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0], label="Cluster ID")

    # Panel 2: Cluster sizes
    cluster_counts = np.bincount(labels)
    axes[1].bar(
        range(len(cluster_counts)),
        cluster_counts,
        color=plt.cm.tab10(np.arange(len(cluster_counts))),
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
    )
    axes[1].set_xlabel("Cluster ID", fontsize=12)
    axes[1].set_ylabel("Size", fontsize=12)
    axes[1].set_title("Cluster Distribution", fontsize=13, fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")

    # Add counts on bars
    for i, count in enumerate(cluster_counts):
        axes[1].text(i, count + 0.2, str(int(count)), ha="center", fontweight="bold")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_fft_speedup(
    vanilla_time: float,
    fft_time: float,
    instance_name: str = "Tai256c",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Compare vanilla vs FFT-accelerated convergence time.

    Parameters
    ----------
    vanilla_time : float
        Time without FFT (seconds)
    fft_time : float
        Time with FFT (seconds)
    instance_name : str
        Instance name for title
    title : str, optional
        Figure title
    save_path : str, optional
        Path to save
    show : bool
        Whether to display

    Returns
    -------
    fig : matplotlib.Figure
    """
    if title is None:
        title = f"FFT-Laplace Acceleration ({instance_name})"

    methods = ["Vanilla", "FFT-Laplace"]
    times = [vanilla_time, fft_time]
    speedup = vanilla_time / fft_time

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart: Times
    bars = ax1.bar(
        methods,
        times,
        color=[COLORS["baseline"], COLORS["novel"]],
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    ax1.set_ylabel("Time (seconds)", fontsize=12)
    ax1.set_title("Convergence Time Comparison", fontsize=13, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    # Annotate bars
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{time_val:.1f}s",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Speedup indicator
    ax2.text(
        0.5,
        0.6,
        f"{speedup:.1f}×",
        fontsize=80,
        ha="center",
        va="center",
        fontweight="bold",
        color=COLORS["novel"],
        transform=ax2.transAxes,
    )
    ax2.text(0.5, 0.25, "Speedup", fontsize=18, ha="center", va="center", transform=ax2.transAxes)
    ax2.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_saddle_timeline(
    history: Dict,
    title: str = "Saddle Point Events Timeline",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot saddle detection and escape events over time.

    Parameters
    ----------
    history : dict
        Keys: 'time', 'objective', 'saddle_detections', 'escapes'
    title : str
        Figure title
    save_path : str, optional
        Path to save
    show : bool
        Whether to display

    Returns
    -------
    fig : matplotlib.Figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: Objective with events
    ax1.plot(
        history["times"],
        history["objectives"],
        color=COLORS["novel"],
        linewidth=2.5,
        label="Objective",
    )

    for t_detect in history.get("saddle_detections", []):
        ax1.axvline(x=t_detect, color=COLORS["accent"], linestyle=":", alpha=0.6, linewidth=1.5)

    for t_escape in history.get("escapes", []):
        ax1.axvline(x=t_escape, color="red", linestyle="--", alpha=0.6, linewidth=1.5)

    ax1.set_ylabel("Objective", fontsize=12)
    ax1.set_title("Saddle Point Events Timeline", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(["Objective"], loc="upper right", fontsize=10)

    # Bottom: Event markers
    detections = history.get("saddle_detections", [])
    escapes = history.get("escapes", [])

    ax2.scatter(
        detections,
        [1] * len(detections),
        s=150,
        c=COLORS["accent"],
        marker="o",
        edgecolors="black",
        linewidth=1.5,
        label="Saddle Detected",
        zorder=5,
    )
    ax2.scatter(
        escapes,
        [2] * len(escapes),
        s=200,
        c="red",
        marker="X",
        edgecolors="black",
        linewidth=1.5,
        label="Escape Triggered",
        zorder=5,
    )

    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_yticks([1, 2])
    ax2.set_yticklabels(["Detect", "Escape"])
    ax2.set_ylim(0.5, 2.5)
    ax2.legend(fontsize=10, loc="upper right")
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_method_effectiveness(
    method_improvements: Dict[str, float],
    title: str = "Method Effectiveness Analysis",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot bar chart of method contributions to overall improvement.

    Parameters
    ----------
    method_improvements : dict
        Keys: method names, Values: % improvement over baseline
    title : str
        Figure title
    save_path : str, optional
        Path to save
    show : bool
        Whether to display

    Returns
    -------
    fig : matplotlib.Figure
    """
    methods = list(method_improvements.keys())
    improvements = list(method_improvements.values())

    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by improvement
    sorted_pairs = sorted(zip(methods, improvements), key=lambda x: x[1], reverse=True)
    methods_sorted, improvements_sorted = zip(*sorted_pairs)

    bars = ax.barh(
        range(len(methods_sorted)),
        improvements_sorted,
        color=COLORS["novel"],
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )

    ax.set_yticks(range(len(methods_sorted)))
    ax.set_yticklabels(methods_sorted)
    ax.set_xlabel("Improvement (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Annotate bars
    for i, (bar, val) in enumerate(zip(bars, improvements_sorted)):
        width = bar.get_width()
        ax.text(
            width + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def create_publication_figure(
    convergence_history: Dict,
    X_init: np.ndarray,
    X_final: np.ndarray,
    P_final: np.ndarray,
    performance_metrics: Dict,
    filename: str = "publication_figure.pdf",
) -> plt.Figure:
    """
    Create publication-quality multi-panel figure suitable for journals.

    6-panel layout (2 rows × 3 columns):
    - Top left: Convergence (objective)
    - Top middle: DS violation
    - Top right: Gradient norm
    - Bottom left: Initial X
    - Bottom middle: Final X
    - Bottom right: Final Permutation

    Parameters
    ----------
    convergence_history : dict
        Convergence data with times, objectives, ds_violations, grad_norms
    X_init : np.ndarray
        Initial doubly-stochastic matrix
    X_final : np.ndarray
        Final doubly-stochastic matrix
    P_final : np.ndarray
        Final permutation matrix
    performance_metrics : dict
        Metrics to display (gap, time, etc.)
    filename : str
        Output filename

    Returns
    -------
    fig : matplotlib.Figure
    """
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

    # Top row: Convergence metrics

    # Panel 1: Objective
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(
        convergence_history["times"],
        convergence_history["objectives"],
        color=COLORS["novel"],
        linewidth=2,
    )
    ax1.set_xlabel("Time (s)", fontsize=10)
    ax1.set_ylabel("Objective", fontsize=10)
    ax1.set_title("(A) Convergence", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Panel 2: DS Violation
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(
        convergence_history["times"],
        convergence_history["ds_violations"],
        color=COLORS["baseline"],
        linewidth=2,
    )
    ax2.set_xlabel("Time (s)", fontsize=10)
    ax2.set_ylabel("DS Violation", fontsize=10)
    ax2.set_title("(B) Feasibility", fontsize=11, fontweight="bold")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3, which="both")

    # Panel 3: Gradient Norm
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(
        convergence_history["times"],
        convergence_history["grad_norms"],
        color=COLORS["neutral"],
        linewidth=2,
    )
    ax3.set_xlabel("Time (s)", fontsize=10)
    ax3.set_ylabel("Gradient Norm", fontsize=10)
    ax3.set_title("(C) Gradient Evolution", fontsize=11, fontweight="bold")
    ax3.set_yscale("log")
    ax3.grid(True, alpha=0.3, which="both")

    # Bottom row: Solution matrices

    # Panel 4: Initial X
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(X_init, cmap="viridis", vmin=0, vmax=1)
    ax4.set_title("(D) Initial X", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Column", fontsize=9)
    ax4.set_ylabel("Row", fontsize=9)

    # Panel 5: Final X
    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(X_final, cmap="viridis", vmin=0, vmax=1)
    ax5.set_title("(E) Final X", fontsize=11, fontweight="bold")
    ax5.set_xlabel("Column", fontsize=9)
    ax5.set_ylabel("Row", fontsize=9)

    # Panel 6: Final Permutation
    ax6 = fig.add_subplot(gs[1, 2])
    im6 = ax6.imshow(P_final, cmap="binary", vmin=0, vmax=1)
    ax6.set_title("(F) Solution P", fontsize=11, fontweight="bold")
    ax6.set_xlabel("Column", fontsize=9)
    ax6.set_ylabel("Row", fontsize=9)

    # Add metrics text box
    metrics_text = f"Gap: {performance_metrics.get('gap', 0):.2f}%\n"
    metrics_text += f"Time: {performance_metrics.get('time', 0):.1f}s\n"
    metrics_text += f"DS Viol: {performance_metrics.get('ds_violation', 0):.2e}"

    fig.text(
        0.5,
        0.02,
        metrics_text,
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.savefig(filename, dpi=300, bbox_inches="tight", format="pdf")
    print(f"[OK] Publication figure saved: {filename}")

    return fig


def create_convergence_animation(
    X_history: List[np.ndarray],
    times: List[float],
    save_path: str = "convergence.gif",
    fps: int = 5,
) -> None:
    """
    Create animated GIF of solution matrix evolution.

    Parameters
    ----------
    X_history : list of np.ndarray
        Snapshots of X matrix at different times
    times : list of float
        Corresponding timestamps
    save_path : str
        Output GIF file path
    fps : int
        Frames per second for animation
    """
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:
        print("[WARNING] matplotlib.animation or Pillow not available, skipping animation")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame):
        ax.clear()
        im = ax.imshow(X_history[frame], cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"t = {times[frame]:.3f}s", fontsize=14, fontweight="bold")
        ax.set_xlabel("Column Index")
        ax.set_ylabel("Row Index")
        plt.colorbar(im, ax=ax, fraction=0.046)
        return [im]

    anim = FuncAnimation(fig, update, frames=len(X_history), interval=1000 // fps, blit=False)

    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer)

    print(f"[OK] Animation saved: {save_path}")
    plt.close(fig)


# Initialize default style on import
load_style()
