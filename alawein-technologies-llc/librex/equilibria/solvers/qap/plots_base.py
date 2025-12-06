"""
Base plotting utilities - Extracted common patterns to eliminate duplication.

This module provides:
- Standard plot setup and formatting
- Common axis configuration
- Publication-quality styling
"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:  # Optional dependency; tests shouldn't require seaborn
    import seaborn as sns
except ModuleNotFoundError:  # pragma: no cover - exercised when seaborn missing
    sns = None

# Color schemes
COLORS = {
    "novel": "#2E7D32",  # Dark green for novel methods
    "baseline": "#C62828",  # Dark red for baseline methods
    "neutral": "#1976D2",  # Blue for neutral/combined
    "accent": "#F57C00",  # Orange for highlights
}

_FALLBACK_COLORBLIND = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]

COLORBLIND_PALETTE = sns.color_palette("colorblind") if sns else _FALLBACK_COLORBLIND


def setup_publication_style():
    """Configure matplotlib for publication-quality plots."""
    if sns:
        sns.set_theme(context="talk", style="whitegrid")

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


def configure_single_axis(
    ax,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    yscale: Optional[str] = None,
    grid: bool = True,
    grid_which: str = "major",
) -> None:
    """
    Configure a single axis with standard styling.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis object to configure
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    title : str, optional
        Axis title
    yscale : str, optional
        Scale for y-axis ('log', 'linear', etc.)
    grid : bool, default=True
        Enable grid
    grid_which : str, default='major'
        Grid lines to show ('major', 'minor', 'both')
    """
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")
    if yscale:
        ax.set_yscale(yscale)
    if grid:
        ax.grid(True, alpha=0.3, which=grid_which)


def save_and_show(
    fig: plt.Figure,
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = 300,
) -> None:
    """
    Save figure and optionally display.

    Parameters
    ----------
    fig : matplotlib.Figure
        Figure to save/show
    save_path : str, optional
        Path to save figure (PDF format inferred)
    show : bool, default=True
        Whether to display figure
    dpi : int, default=300
        Resolution for saved figure
    """
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()


def create_figure_with_subplots(
    rows: int = 1, cols: int = 1, figsize: Optional[Tuple[int, int]] = None, **kwargs
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create figure with standard sizing and layout.

    Parameters
    ----------
    rows : int, default=1
        Number of rows
    cols : int, default=1
        Number of columns
    figsize : tuple of int, optional
        Figure size (width, height). If None, auto-sized.
    **kwargs
        Additional arguments to plt.subplots()

    Returns
    -------
    fig : matplotlib.Figure
        Figure object
    axes : np.ndarray of Axes
        Subplot axes (1D array if single row/col, 2D otherwise)
    """
    if figsize is None:
        figsize = (5 * cols, 4 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, **kwargs)

    # Ensure axes is always an array (even for single subplot)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(-1, 1) if cols == 1 else axes.reshape(1, -1)

    return fig, axes


def add_gate_line(
    ax,
    gate_value: float,
    label: Optional[str] = None,
    color: str = "red",
    linestyle: str = "--",
) -> None:
    """
    Add horizontal gate line to plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to add line to
    gate_value : float
        Y-value of gate line
    label : str, optional
        Legend label for gate
    color : str, default='red'
        Line color
    linestyle : str, default='--'
        Line style
    """
    if label is None:
        label = f"Gate ({gate_value})"

    ax.axhline(y=gate_value, color=color, linestyle=linestyle, linewidth=2, label=label)


def add_best_indicator(
    ax,
    value: float,
    label: str,
    color: str = "green",
    linestyle: str = "-",
) -> None:
    """
    Add vertical line indicating best value.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to add line to
    value : float
        X-value of indicator
    label : str
        Legend label
    color : str, default='green'
        Line color
    linestyle : str, default='-'
        Line style
    """
    ax.axvline(x=value, color=color, linestyle=linestyle, linewidth=2, label=label)


__all__ = [
    "COLORS",
    "COLORBLIND_PALETTE",
    "setup_publication_style",
    "configure_single_axis",
    "save_and_show",
    "create_figure_with_subplots",
    "add_gate_line",
    "add_best_indicator",
]
