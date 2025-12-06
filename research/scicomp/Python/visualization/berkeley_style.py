#!/usr/bin/env python3
"""
Berkeley Visual Identity - Scientific Plotting Style
Official UC Berkeley color scheme and styling for publication-quality figures.
Implements university branding guidelines for academic and research publications.
Colors follow UC Berkeley's official brand guidelines:
- Primary: Berkeley Blue (#003262), California Gold (#FDB515)
- Secondary palette for diverse visualizations
- High-contrast accessibility compliance
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
import seaborn as sns
from pathlib import Path
# UC Berkeley Official Color Palette
BERKELEY_COLORS = {
    'primary': {
        'berkeley_blue': '#003262',
        'california_gold': '#FDB515'
    },
    'secondary': {
        'blue_dark': '#010133',
        'gold_dark': '#FC9313',
        'green_dark': '#00553A',
        'rose_dark': '#770747',
        'purple_dark': '#431170',
        'red_dark': '#8C1515',
        'orange_dark': '#D2691E',
        'teal_dark': '#004C5A'
    },
    'neutral': {
        'grey_light': '#D9D9D9',
        'grey_medium': '#999999',
        'grey_dark': '#666666',
        'black': '#000000',
        'white': '#FFFFFF'
    }
}
# Flattened color lists for easy access
BERKELEY_BLUE = BERKELEY_COLORS['primary']['berkeley_blue']
CALIFORNIA_GOLD = BERKELEY_COLORS['primary']['california_gold']
# Color sequences for multi-line plots
BERKELEY_SEQUENCE = [
    BERKELEY_BLUE,
    CALIFORNIA_GOLD,
    BERKELEY_COLORS['secondary']['green_dark'],
    BERKELEY_COLORS['secondary']['rose_dark'],
    BERKELEY_COLORS['secondary']['purple_dark'],
    BERKELEY_COLORS['secondary']['red_dark'],
    BERKELEY_COLORS['secondary']['orange_dark'],
    BERKELEY_COLORS['secondary']['teal_dark']
]
# Plot style configuration
BERKELEY_STYLE = {
    'figure.figsize': (10, 6),
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    # Colors
    'axes.prop_cycle': plt.cycler('color', BERKELEY_SEQUENCE),
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes.edgecolor': BERKELEY_COLORS['neutral']['black'],
    'axes.linewidth': 1.5,
    # Fonts
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 18,
    # Ticks
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.color': BERKELEY_COLORS['neutral']['black'],
    'ytick.color': BERKELEY_COLORS['neutral']['black'],
    # Lines and markers
    'lines.linewidth': 2.0,
    'lines.markersize': 8,
    'patch.linewidth': 1.0,
    # Grid
    'axes.grid': False,  # Berkeley style avoids grids unless needed
    'grid.color': BERKELEY_COLORS['neutral']['grey_light'],
    'grid.linewidth': 0.5,
    'grid.alpha': 0.8,
    # Legend
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.shadow': False,
    'legend.framealpha': 1.0,
    'legend.facecolor': 'white',
    'legend.edgecolor': BERKELEY_COLORS['neutral']['black'],
    'legend.borderpad': 0.4,
    'legend.columnspacing': 1.0,
    'legend.handlelength': 1.5,
}
def apply_berkeley_style():
    """Apply Berkeley visual identity to matplotlib."""
    plt.style.use('default')  # Reset to default first
    mpl.rcParams.update(BERKELEY_STYLE)
    # Register custom colormaps
    _register_berkeley_colormaps()
def _register_berkeley_colormaps():
    """Register Berkeley-themed colormaps."""
    # Blue-Gold gradient
    blue_gold_colors = [BERKELEY_BLUE, CALIFORNIA_GOLD]
    blue_gold_cmap = LinearSegmentedColormap.from_list('berkeley_blue_gold', blue_gold_colors)
    plt.cm.register_cmap(name='berkeley_blue_gold', cmap=blue_gold_cmap)
    # Full spectrum using Berkeley colors
    spectrum_colors = [
        BERKELEY_BLUE,
        BERKELEY_COLORS['secondary']['purple_dark'],
        BERKELEY_COLORS['secondary']['rose_dark'],
        BERKELEY_COLORS['secondary']['red_dark'],
        BERKELEY_COLORS['secondary']['orange_dark'],
        CALIFORNIA_GOLD
    ]
    spectrum_cmap = LinearSegmentedColormap.from_list('berkeley_spectrum', spectrum_colors)
    plt.cm.register_cmap(name='berkeley_spectrum', cmap=spectrum_cmap)
    # Sequential blue
    blue_colors = ['white', BERKELEY_BLUE]
    blue_cmap = LinearSegmentedColormap.from_list('berkeley_blues', blue_colors)
    plt.cm.register_cmap(name='berkeley_blues', cmap=blue_cmap)
class BerkeleyPlot:
    """
    Berkeley-styled plotting class for scientific figures.
    Provides convenient methods for creating publication-quality plots
    following UC Berkeley's visual identity guidelines.
    """
    def __init__(self, figsize: Tuple[float, float] = (10, 6),
                 style: str = 'publication'):
        """
        Initialize Berkeley-styled plot.
        Parameters
        ----------
        figsize : tuple, default (10, 6)
            Figure size in inches
        style : str, default 'publication'
            Plot style ('publication', 'presentation', 'poster')
        """
        self.figsize = figsize
        self.style = style
        # Apply Berkeley styling
        apply_berkeley_style()
        # Adjust for different contexts
        if style == 'presentation':
            mpl.rcParams['font.size'] = 14
            mpl.rcParams['axes.titlesize'] = 18
            mpl.rcParams['axes.labelsize'] = 14
        elif style == 'poster':
            mpl.rcParams['font.size'] = 16
            mpl.rcParams['axes.titlesize'] = 22
            mpl.rcParams['axes.labelsize'] = 18
            mpl.rcParams['lines.linewidth'] = 3.0
            mpl.rcParams['lines.markersize'] = 10
        self.fig = None
        self.axes = None
    def create_figure(self, nrows: int = 1, ncols: int = 1,
                     subplot_kw: Optional[Dict] = None,
                     gridspec_kw: Optional[Dict] = None) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
        """
        Create Berkeley-styled figure and axes.
        Parameters
        ----------
        nrows, ncols : int, default 1
            Number of subplot rows and columns
        subplot_kw : dict, optional
            Additional subplot keywords
        gridspec_kw : dict, optional
            GridSpec keywords
        Returns
        -------
        fig : Figure
            Matplotlib figure
        axes : Axes or array of Axes
            Subplot axes
        """
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=self.figsize,
                                          subplot_kw=subplot_kw,
                                          gridspec_kw=gridspec_kw)
        return self.fig, self.axes
    def line_plot(self, x: np.ndarray, y: Union[np.ndarray, List[np.ndarray]],
                 labels: Optional[List[str]] = None,
                 colors: Optional[List[str]] = None,
                 linestyles: Optional[List[str]] = None,
                 markers: Optional[List[str]] = None,
                 title: Optional[str] = None,
                 xlabel: Optional[str] = None,
                 ylabel: Optional[str] = None,
                 legend: bool = True,
                 ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Create Berkeley-styled line plot.
        Parameters
        ----------
        x : ndarray
            X-axis data
        y : ndarray or list of ndarray
            Y-axis data (single array or list for multiple lines)
        labels : list of str, optional
            Line labels for legend
        colors : list of str, optional
            Line colors (uses Berkeley sequence if None)
        linestyles : list of str, optional
            Line styles
        markers : list of str, optional
            Marker styles
        title : str, optional
            Plot title
        xlabel, ylabel : str, optional
            Axis labels
        legend : bool, default True
            Whether to show legend
        ax : Axes, optional
            Matplotlib axes to plot on
        Returns
        -------
        Axes
            Matplotlib axes object
        """
        if ax is None:
            if self.fig is None:
                self.create_figure()
            ax = self.axes if not isinstance(self.axes, np.ndarray) else self.axes[0]
        # Handle single vs multiple y arrays
        if isinstance(y, np.ndarray) and y.ndim == 1:
            y = [y]
        elif isinstance(y, list) and not isinstance(y[0], np.ndarray):
            y = [np.array(y)]
        # Default parameters
        n_lines = len(y)
        if colors is None:
            colors = BERKELEY_SEQUENCE[:n_lines]
        if linestyles is None:
            linestyles = ['-'] * n_lines
        if markers is None:
            markers = [''] * n_lines
        if labels is None:
            labels = [f'Line {i+1}' for i in range(n_lines)]
        # Plot lines
        for i, y_data in enumerate(y):
            ax.plot(x, y_data,
                   color=colors[i % len(colors)],
                   linestyle=linestyles[i % len(linestyles)],
                   marker=markers[i % len(markers)],
                   label=labels[i],
                   linewidth=mpl.rcParams['lines.linewidth'],
                   markersize=mpl.rcParams['lines.markersize'])
        # Formatting
        if title:
            ax.set_title(title, fontweight='bold', color=BERKELEY_BLUE)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if legend and len(y) > 1:
            ax.legend()
        return ax
    def scatter_plot(self, x: np.ndarray, y: np.ndarray,
                    c: Optional[np.ndarray] = None,
                    s: Optional[Union[float, np.ndarray]] = None,
                    alpha: float = 0.7,
                    cmap: str = 'berkeley_spectrum',
                    title: Optional[str] = None,
                    xlabel: Optional[str] = None,
                    ylabel: Optional[str] = None,
                    colorbar: bool = True,
                    ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Create Berkeley-styled scatter plot.
        Parameters
        ----------
        x, y : ndarray
            Scatter plot coordinates
        c : ndarray, optional
            Color values for points
        s : float or ndarray, optional
            Point sizes
        alpha : float, default 0.7
            Point transparency
        cmap : str, default 'berkeley_spectrum'
            Colormap name
        title : str, optional
            Plot title
        xlabel, ylabel : str, optional
            Axis labels
        colorbar : bool, default True
            Whether to show colorbar
        ax : Axes, optional
            Matplotlib axes to plot on
        Returns
        -------
        Axes
            Matplotlib axes object
        """
        if ax is None:
            if self.fig is None:
                self.create_figure()
            ax = self.axes if not isinstance(self.axes, np.ndarray) else self.axes[0]
        # Default point size
        if s is None:
            s = mpl.rcParams['lines.markersize']**2
        # Default color
        if c is None:
            c = BERKELEY_BLUE
            colorbar = False
        # Create scatter plot
        scatter = ax.scatter(x, y, c=c, s=s, alpha=alpha, cmap=cmap)
        # Add colorbar if requested and c is array
        if colorbar and isinstance(c, np.ndarray):
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.ax.tick_params(direction='in')
        # Formatting
        if title:
            ax.set_title(title, fontweight='bold', color=BERKELEY_BLUE)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        return ax
    def heatmap(self, data: np.ndarray,
               x_labels: Optional[List[str]] = None,
               y_labels: Optional[List[str]] = None,
               cmap: str = 'berkeley_blues',
               title: Optional[str] = None,
               xlabel: Optional[str] = None,
               ylabel: Optional[str] = None,
               colorbar: bool = True,
               ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Create Berkeley-styled heatmap.
        Parameters
        ----------
        data : ndarray
            2D data array
        x_labels, y_labels : list of str, optional
            Axis tick labels
        cmap : str, default 'berkeley_blues'
            Colormap name
        title : str, optional
            Plot title
        xlabel, ylabel : str, optional
            Axis labels
        colorbar : bool, default True
            Whether to show colorbar
        ax : Axes, optional
            Matplotlib axes to plot on
        Returns
        -------
        Axes
            Matplotlib axes object
        """
        if ax is None:
            if self.fig is None:
                self.create_figure()
            ax = self.axes if not isinstance(self.axes, np.ndarray) else self.axes[0]
        # Create heatmap
        im = ax.imshow(data, cmap=cmap, aspect='auto', origin='lower')
        # Set labels
        if x_labels is not None:
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels)
        if y_labels is not None:
            ax.set_yticks(range(len(y_labels)))
            ax.set_yticklabels(y_labels)
        # Add colorbar
        if colorbar:
            cbar = plt.colorbar(im, ax=ax)
            cbar.ax.tick_params(direction='in')
        # Formatting
        if title:
            ax.set_title(title, fontweight='bold', color=BERKELEY_BLUE)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        return ax
    def wavefunction(self, x: np.ndarray, psi: np.ndarray,
                    title: Optional[str] = None,
                    show_probability: bool = True,
                    ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot quantum wavefunction with Berkeley styling.
        Parameters
        ----------
        x : ndarray
            Position coordinates
        psi : ndarray
            Wavefunction values (can be complex)
        title : str, optional
            Plot title
        show_probability : bool, default True
            Whether to show probability density
        ax : Axes, optional
            Matplotlib axes to plot on
        Returns
        -------
        Axes
            Matplotlib axes object
        """
        if ax is None:
            if self.fig is None:
                self.create_figure()
            ax = self.axes if not isinstance(self.axes, np.ndarray) else self.axes[0]
        # Plot real and imaginary parts
        ax.plot(x, np.real(psi), color=BERKELEY_BLUE, linewidth=2,
               label='Re[ψ(x)]')
        if np.any(np.imag(psi) != 0):
            ax.plot(x, np.imag(psi), color=CALIFORNIA_GOLD, linewidth=2,
                   linestyle='--', label='Im[ψ(x)]')
        # Probability density
        if show_probability:
            ax.fill_between(x, np.abs(psi)**2, alpha=0.3,
                           color=BERKELEY_COLORS['secondary']['green_dark'],
                           label='|ψ(x)|²')
        # Formatting
        ax.set_xlabel('Position')
        ax.set_ylabel('Wavefunction')
        ax.legend()
        ax.grid(True, alpha=0.3)
        if title:
            ax.set_title(title, fontweight='bold', color=BERKELEY_BLUE)
        return ax
    def save_figure(self, filename: Union[str, Path],
                   dpi: int = 300,
                   format: str = 'png',
                   bbox_inches: str = 'tight',
                   pad_inches: float = 0.1,
                   transparent: bool = False) -> None:
        """
        Save figure with Berkeley styling.
        Parameters
        ----------
        filename : str or Path
            Output filename
        dpi : int, default 300
            Resolution in dots per inch
        format : str, default 'png'
            Output format
        bbox_inches : str, default 'tight'
            Bounding box setting
        pad_inches : float, default 0.1
            Padding around figure
        transparent : bool, default False
            Whether background should be transparent
        """
        if self.fig is None:
            raise ValueError("No figure to save. Create a figure first.")
        self.fig.savefig(filename, dpi=dpi, format=format,
                        bbox_inches=bbox_inches, pad_inches=pad_inches,
                        transparent=transparent, facecolor='white')
# Convenience functions
def publication_figure(figsize: Tuple[float, float] = (10, 6)) -> BerkeleyPlot:
    """Create publication-ready Berkeley-styled figure."""
    return BerkeleyPlot(figsize=figsize, style='publication')
def presentation_figure(figsize: Tuple[float, float] = (12, 8)) -> BerkeleyPlot:
    """Create presentation-ready Berkeley-styled figure."""
    return BerkeleyPlot(figsize=figsize, style='presentation')
def poster_figure(figsize: Tuple[float, float] = (16, 10)) -> BerkeleyPlot:
    """Create poster-ready Berkeley-styled figure."""
    return BerkeleyPlot(figsize=figsize, style='poster')
def save_publication_figure(fig: plt.Figure, filename: Union[str, Path],
                          dpi: int = 300, format: str = 'png') -> None:
    """
    Save publication-quality figure with proper settings.
    Parameters
    ----------
    fig : Figure
        Matplotlib figure to save
    filename : str or Path
        Output filename
    dpi : int, default 300
        Resolution
    format : str, default 'png'
        Output format
    """
    fig.savefig(filename, dpi=dpi, format=format, bbox_inches='tight',
               pad_inches=0.1, facecolor='white', edgecolor='none')
# Initialize Berkeley style on import
apply_berkeley_style()