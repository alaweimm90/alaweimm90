"""
Scientific Plotting Module
=========================
Professional scientific plotting with Berkeley branding and publication-ready formatting.
Author: Berkeley SciComp Team
Date: 2024
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
from typing import Optional, List, Tuple, Dict, Any, Union
import warnings
# Berkeley color scheme
BERKELEY_BLUE = '#003262'
CALIFORNIA_GOLD = '#FDB515'
BERKELEY_LIGHT_BLUE = '#3B7EA1'
BERKELEY_PALETTE = [
    BERKELEY_BLUE,
    CALIFORNIA_GOLD,
    BERKELEY_LIGHT_BLUE,
    '#C3282C',  # Berkeley Red
    '#7B2142',  # Berkeley Maroon
    '#007C7C',  # Berkeley Teal
    '#A4A4A4',  # Berkeley Medium Gray
    '#46535C'   # Berkeley Dark Gray
]
class ScientificPlot:
    """
    Professional scientific plotting class with Berkeley branding.
    Provides publication-ready plots with consistent styling,
    proper error handling, and scientific formatting conventions.
    """
    def __init__(self, style: str = 'berkeley', dpi: int = 300):
        """
        Initialize scientific plot with Berkeley styling.
        Args:
            style: Plot style ('berkeley', 'minimal', 'presentation')
            dpi: Resolution for saved figures
        """
        self.style = style
        self.dpi = dpi
        self.colors = BERKELEY_PALETTE.copy()
        self.color_index = 0
        # Set matplotlib parameters
        self._setup_style()
    def _setup_style(self):
        """Configure matplotlib parameters for scientific plotting."""
        plt.rcParams.update({
            'figure.dpi': self.dpi,
            'savefig.dpi': self.dpi,
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'font.family': ['DejaVu Sans', 'Arial', 'sans-serif'],
            'axes.linewidth': 1.2,
            'xtick.major.width': 1.0,
            'ytick.major.width': 1.0,
            'xtick.minor.width': 0.5,
            'ytick.minor.width': 0.5,
            'lines.linewidth': 2.0,
            'lines.markersize': 6,
            'axes.grid': False,
            'grid.alpha': 0.3,
            'axes.axisbelow': True
        })
        if self.style == 'berkeley':
            plt.rcParams.update({
                'axes.prop_cycle': plt.cycler('color', BERKELEY_PALETTE),
                'figure.facecolor': 'white',
                'axes.facecolor': 'white',
                'axes.edgecolor': BERKELEY_BLUE,
                'axes.labelcolor': BERKELEY_BLUE,
                'text.color': BERKELEY_BLUE,
                'xtick.color': BERKELEY_BLUE,
                'ytick.color': BERKELEY_BLUE
            })
    def create_figure(self, figsize: Tuple[float, float] = (10, 6),
                     nrows: int = 1, ncols: int = 1) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
        """
        Create a new figure with Berkeley styling.
        Args:
            figsize: Figure size in inches
            nrows: Number of subplot rows
            ncols: Number of subplot columns
        Returns:
            Tuple of (figure, axes)
        """
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        # Add Berkeley branding
        if self.style == 'berkeley':
            fig.patch.set_facecolor('white')
            # Add subtle Berkeley logo/branding
            if nrows == 1 and ncols == 1:
                self._add_berkeley_watermark(fig, axes)
        return fig, axes
    def _add_berkeley_watermark(self, fig: plt.Figure, ax: plt.Axes):
        """Add subtle Berkeley branding to the plot."""
        # Add a very subtle watermark
        fig.text(0.99, 0.01, 'Berkeley SciComp',
                fontsize=8, color=BERKELEY_BLUE, alpha=0.3,
                ha='right', va='bottom', style='italic')
    def plot(self, x: np.ndarray, y: np.ndarray,
             ax: Optional[plt.Axes] = None,
             label: Optional[str] = None,
             color: Optional[str] = None,
             style: str = '-',
             marker: Optional[str] = None,
             **kwargs) -> plt.Line2D:
        """
        Create a line plot with Berkeley styling.
        Args:
            x: X-axis data
            y: Y-axis data
            ax: Axes to plot on (current axes if None)
            label: Data series label
            color: Line color (auto-select if None)
            style: Line style
            marker: Marker style
            **kwargs: Additional plot parameters
        Returns:
            Line2D object
        """
        if ax is None:
            ax = plt.gca()
        if color is None:
            color = self._get_next_color()
        line = ax.plot(x, y, style, color=color, label=label,
                      marker=marker, **kwargs)[0]
        return line
    def scatter(self, x: np.ndarray, y: np.ndarray,
                ax: Optional[plt.Axes] = None,
                label: Optional[str] = None,
                color: Optional[str] = None,
                size: Union[float, np.ndarray] = 50,
                alpha: float = 0.7,
                **kwargs) -> plt.PathCollection:
        """
        Create a scatter plot with Berkeley styling.
        Args:
            x: X-axis data
            y: Y-axis data
            ax: Axes to plot on
            label: Data series label
            color: Point color
            size: Point size(s)
            alpha: Transparency
            **kwargs: Additional scatter parameters
        Returns:
            PathCollection object
        """
        if ax is None:
            ax = plt.gca()
        if color is None:
            color = self._get_next_color()
        scatter = ax.scatter(x, y, s=size, c=color, alpha=alpha,
                           label=label, **kwargs)
        return scatter
    def errorbar(self, x: np.ndarray, y: np.ndarray,
                 yerr: Optional[np.ndarray] = None,
                 xerr: Optional[np.ndarray] = None,
                 ax: Optional[plt.Axes] = None,
                 label: Optional[str] = None,
                 color: Optional[str] = None,
                 **kwargs) -> plt.ErrorbarContainer:
        """
        Create an error bar plot.
        Args:
            x: X-axis data
            y: Y-axis data
            yerr: Y-axis error bars
            xerr: X-axis error bars
            ax: Axes to plot on
            label: Data series label
            color: Plot color
            **kwargs: Additional errorbar parameters
        Returns:
            ErrorbarContainer object
        """
        if ax is None:
            ax = plt.gca()
        if color is None:
            color = self._get_next_color()
        errorbar = ax.errorbar(x, y, yerr=yerr, xerr=xerr,
                              color=color, label=label,
                              capsize=3, capthick=1, **kwargs)
        return errorbar
    def fill_between(self, x: np.ndarray, y1: np.ndarray, y2: np.ndarray = 0,
                     ax: Optional[plt.Axes] = None,
                     label: Optional[str] = None,
                     color: Optional[str] = None,
                     alpha: float = 0.3,
                     **kwargs) -> plt.PolyCollection:
        """
        Fill area between curves.
        Args:
            x: X-axis data
            y1: Upper boundary
            y2: Lower boundary
            ax: Axes to plot on
            label: Fill label
            color: Fill color
            alpha: Transparency
            **kwargs: Additional fill parameters
        Returns:
            PolyCollection object
        """
        if ax is None:
            ax = plt.gca()
        if color is None:
            color = self._get_next_color()
        fill = ax.fill_between(x, y1, y2, color=color, alpha=alpha,
                              label=label, **kwargs)
        return fill
    def set_labels(self, xlabel: str, ylabel: str, title: str = '',
                   ax: Optional[plt.Axes] = None):
        """
        Set axis labels and title with Berkeley styling.
        Args:
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Plot title
            ax: Axes to modify
        """
        if ax is None:
            ax = plt.gca()
        ax.set_xlabel(xlabel, color=BERKELEY_BLUE, fontweight='bold')
        ax.set_ylabel(ylabel, color=BERKELEY_BLUE, fontweight='bold')
        if title:
            ax.set_title(title, color=BERKELEY_BLUE, fontweight='bold', pad=20)
    def add_legend(self, ax: Optional[plt.Axes] = None,
                   location: str = 'best', **kwargs):
        """
        Add a professionally styled legend.
        Args:
            ax: Axes to add legend to
            location: Legend location
            **kwargs: Additional legend parameters
        """
        if ax is None:
            ax = plt.gca()
        legend = ax.legend(loc=location, frameon=True,
                          fancybox=True, shadow=True, **kwargs)
        # Style the legend
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor(BERKELEY_BLUE)
        legend.get_frame().set_alpha(0.9)
        return legend
    def add_grid(self, ax: Optional[plt.Axes] = None,
                 alpha: float = 0.3, which: str = 'major'):
        """
        Add a professional grid.
        Args:
            ax: Axes to add grid to
            alpha: Grid transparency
            which: Grid type ('major', 'minor', 'both')
        """
        if ax is None:
            ax = plt.gca()
        ax.grid(True, alpha=alpha, which=which, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
    def add_inset(self, ax: plt.Axes, bounds: List[float],
                  xlim: Tuple[float, float], ylim: Tuple[float, float]) -> plt.Axes:
        """
        Add an inset plot.
        Args:
            ax: Parent axes
            bounds: Inset bounds [x, y, width, height] in parent coordinates
            xlim: Inset x-axis limits
            ylim: Inset y-axis limits
        Returns:
            Inset axes
        """
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        inset_ax = inset_axes(ax, width="30%", height="30%",
                             bbox_to_anchor=bounds, bbox_transform=ax.transAxes)
        inset_ax.set_xlim(xlim)
        inset_ax.set_ylim(ylim)
        # Style inset
        inset_ax.tick_params(labelsize=8)
        return inset_ax
    def add_annotation(self, text: str, xy: Tuple[float, float],
                      xytext: Tuple[float, float],
                      ax: Optional[plt.Axes] = None,
                      **kwargs):
        """
        Add an annotation with arrow.
        Args:
            text: Annotation text
            xy: Point to annotate
            xytext: Text position
            ax: Axes to annotate
            **kwargs: Additional annotation parameters
        """
        if ax is None:
            ax = plt.gca()
        default_props = {
            'arrowprops': dict(arrowstyle='->', color=BERKELEY_BLUE, lw=1.5),
            'fontsize': 10,
            'color': BERKELEY_BLUE,
            'bbox': dict(boxstyle="round,pad=0.3", facecolor='white',
                        edgecolor=BERKELEY_BLUE, alpha=0.8)
        }
        # Merge with user kwargs
        props = {**default_props, **kwargs}
        ax.annotate(text, xy=xy, xytext=xytext, **props)
    def save_figure(self, filename: str, fig: Optional[plt.Figure] = None,
                    **kwargs):
        """
        Save figure with high quality settings.
        Args:
            filename: Output filename
            fig: Figure to save (current figure if None)
            **kwargs: Additional savefig parameters
        """
        if fig is None:
            fig = plt.gcf()
        default_params = {
            'dpi': self.dpi,
            'bbox_inches': 'tight',
            'facecolor': 'white',
            'edgecolor': 'none',
            'transparent': False
        }
        params = {**default_params, **kwargs}
        fig.savefig(filename, **params)
    def _get_next_color(self) -> str:
        """Get the next color in the Berkeley palette."""
        color = self.colors[self.color_index % len(self.colors)]
        self.color_index += 1
        return color
    def reset_colors(self):
        """Reset color cycling."""
        self.color_index = 0
class Plot3D:
    """
    3D scientific plotting with Berkeley styling.
    """
    def __init__(self, style: str = 'berkeley'):
        """Initialize 3D plot with Berkeley styling."""
        self.style = style
        self.colors = BERKELEY_PALETTE.copy()
        self.color_index = 0
    def create_figure(self, figsize: Tuple[float, float] = (10, 8)) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a 3D figure.
        Args:
            figsize: Figure size
        Returns:
            Tuple of (figure, 3D axes)
        """
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        # Berkeley styling
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(BERKELEY_BLUE)
        ax.yaxis.pane.set_edgecolor(BERKELEY_BLUE)
        ax.zaxis.pane.set_edgecolor(BERKELEY_BLUE)
        return fig, ax
    def surface(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                ax: plt.Axes, colormap: str = 'viridis', alpha: float = 0.8,
                **kwargs):
        """
        Create a 3D surface plot.
        Args:
            X: X-axis meshgrid
            Y: Y-axis meshgrid
            Z: Surface heights
            ax: 3D axes
            colormap: Color map name
            alpha: Surface transparency
            **kwargs: Additional surface parameters
        """
        from matplotlib import cm
        surface = ax.plot_surface(X, Y, Z, cmap=colormap, alpha=alpha, **kwargs)
        return surface
    def scatter3d(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                  ax: plt.Axes, color: Optional[str] = None, size: float = 50,
                  **kwargs):
        """
        Create a 3D scatter plot.
        Args:
            x: X-axis data
            y: Y-axis data
            z: Z-axis data
            ax: 3D axes
            color: Point color
            size: Point size
            **kwargs: Additional scatter parameters
        """
        if color is None:
            color = self._get_next_color()
        scatter = ax.scatter(x, y, z, c=color, s=size, **kwargs)
        return scatter
    def _get_next_color(self) -> str:
        """Get the next color in the Berkeley palette."""
        color = self.colors[self.color_index % len(self.colors)]
        self.color_index += 1
        return color
def demo():
    """Demonstrate scientific plotting capabilities."""
    import numpy as np
    print("Berkeley SciComp - Scientific Plotting Demo")
    print("=" * 50)
    # Generate sample data
    x = np.linspace(0, 4*np.pi, 200)
    y1 = np.sin(x) * np.exp(-x/8)
    y2 = np.cos(x) * np.exp(-x/8)
    y3 = np.sin(2*x) * np.exp(-x/6)
    # Create scientific plot
    sci_plot = ScientificPlot()
    fig, ax = sci_plot.create_figure(figsize=(12, 8))
    # Plot multiple series
    sci_plot.plot(x, y1, label='Damped Sine', color=BERKELEY_BLUE)
    sci_plot.plot(x, y2, label='Damped Cosine', color=CALIFORNIA_GOLD)
    sci_plot.plot(x, y3, label='Double Frequency', color=BERKELEY_LIGHT_BLUE,
                 style='--', marker='o', markersize=3, markevery=10)
    # Add error bars for one series
    x_err = x[::20]
    y_err = y1[::20]
    err = 0.05 * np.abs(y_err)
    sci_plot.errorbar(x_err, y_err, yerr=err, color=BERKELEY_BLUE,
                     alpha=0.7, linestyle='none', marker='s', markersize=4)
    # Fill between curves
    sci_plot.fill_between(x, y1, y2, alpha=0.2, color=BERKELEY_LIGHT_BLUE)
    # Set labels and formatting
    sci_plot.set_labels('Time (s)', 'Amplitude', 'Damped Oscillations Study')
    sci_plot.add_legend(location='upper right')
    sci_plot.add_grid(alpha=0.3)
    # Add annotation
    max_idx = np.argmax(y1)
    sci_plot.add_annotation(f'Maximum: ({x[max_idx]:.2f}, {y1[max_idx]:.2f})',
                           xy=(x[max_idx], y1[max_idx]),
                           xytext=(x[max_idx] + 2, y1[max_idx] + 0.2))
    plt.tight_layout()
    plt.show()
    # 3D surface plot demo
    print("\nCreating 3D surface plot...")
    plot3d = Plot3D()
    fig3d, ax3d = plot3d.create_figure()
    # Generate 3D data
    x3d = np.linspace(-2, 2, 50)
    y3d = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x3d, y3d)
    Z = np.sin(np.sqrt(X**2 + Y**2)) * np.exp(-0.1*(X**2 + Y**2))
    surface = plot3d.surface(X, Y, Z, ax3d, colormap='coolwarm')
    ax3d.set_xlabel('X', color=BERKELEY_BLUE, fontweight='bold')
    ax3d.set_ylabel('Y', color=BERKELEY_BLUE, fontweight='bold')
    ax3d.set_zlabel('Z', color=BERKELEY_BLUE, fontweight='bold')
    ax3d.set_title('3D Surface: Damped Wave Function',
                   color=BERKELEY_BLUE, fontweight='bold')
    plt.colorbar(surface, shrink=0.5)
    plt.show()
    print("Scientific plotting demo completed!")
if __name__ == "__main__":
    demo()