"""Berkeley-themed Visualization for Monte Carlo Methods.
This module provides comprehensive visualization utilities for Monte Carlo
simulations with UC Berkeley's official branding and color scheme.
Classes:
    MonteCarloVisualizer: Main visualization class
Functions:
    plot_convergence: Plot convergence history
    plot_samples: Plot sample distributions
    plot_distributions: Plot probability distributions
    plot_sensitivity: Plot sensitivity analysis results
Colors:
    Berkeley Blue: #003262
    California Gold: #FDB515
Author: Berkeley SciComp Team
Date: 2024
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
from typing import Union, List, Dict, Optional, Tuple, Any
import warnings
from .utils import compute_statistics
from .uncertainty import UncertaintyResult, SensitivityResult
from .sampling import SamplingResult
from .integration import IntegrationResult
from .optimization import OptimizationResult
# Berkeley Color Scheme
BERKELEY_COLORS = {
    'berkeley_blue': '#003262',
    'california_gold': '#FDB515',
    'founders_rock': '#3B7EA1',
    'medalist': '#C4820E',
    'bay_fog': '#DDD5C7',
    'lawrence': '#00B0DA',
    'lap_lane': '#00A598',
    'sather_gate': '#B9D3B6',
    'ion': '#CFDD45',
    'golden_gate': '#ED4E33',
    'rose_garden': '#EE1F60',
    'wellman_tile': '#D9661F'
}
# Plotting parameters
BERKELEY_STYLE = {
    'figure.figsize': [10, 6],
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': BERKELEY_COLORS['berkeley_blue'],
    'axes.linewidth': 1.5,
    'axes.grid': True,
    'axes.axisbelow': True,
    'grid.color': BERKELEY_COLORS['bay_fog'],
    'grid.linestyle': '-',
    'grid.linewidth': 0.5,
    'xtick.color': BERKELEY_COLORS['berkeley_blue'],
    'ytick.color': BERKELEY_COLORS['berkeley_blue'],
    'text.color': BERKELEY_COLORS['berkeley_blue'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 2,
    'lines.markersize': 6
}
class MonteCarloVisualizer:
    """Comprehensive Monte Carlo visualization toolkit.
    Provides publication-ready plots for all Monte Carlo analysis results
    with consistent Berkeley branding.
    """
    def __init__(self, style: str = 'berkeley', figsize: Tuple[float, float] = (10, 6)):
        self.style = style
        self.figsize = figsize
        self.colors = BERKELEY_COLORS.copy()
        # Apply Berkeley style
        if style == 'berkeley':
            plt.style.use('default')  # Reset first
            plt.rcParams.update(BERKELEY_STYLE)
    def plot_convergence(self,
                        convergence_data: Union[List[float], Dict[str, List[float]]],
                        title: str = "Monte Carlo Convergence",
                        xlabel: str = "Iteration",
                        ylabel: str = "Value",
                        true_value: Optional[float] = None,
                        confidence_bands: Optional[Dict[str, List[float]]] = None,
                        log_scale: bool = False,
                        save_path: Optional[str] = None,
                        **kwargs) -> plt.Figure:
        """Plot convergence history for Monte Carlo methods.
        Args:
            convergence_data: Convergence history (single series or multiple)
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            true_value: Reference true value
            confidence_bands: Confidence interval data
            log_scale: Use logarithmic y-axis
            save_path: Path to save figure
            **kwargs: Additional plot arguments
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        # Handle single series vs multiple series
        if isinstance(convergence_data, list):
            convergence_data = {'Main': convergence_data}
        # Color cycle
        color_cycle = [
            self.colors['berkeley_blue'],
            self.colors['california_gold'],
            self.colors['founders_rock'],
            self.colors['medalist'],
            self.colors['lawrence']
        ]
        for i, (label, data) in enumerate(convergence_data.items()):
            color = color_cycle[i % len(color_cycle)]
            x_data = range(1, len(data) + 1)
            ax.plot(x_data, data, label=label, color=color, linewidth=2)
            # Add confidence bands if provided
            if confidence_bands and label in confidence_bands:
                lower = confidence_bands[label]['lower']
                upper = confidence_bands[label]['upper']
                ax.fill_between(x_data, lower, upper, alpha=0.3, color=color)
        # Add true value line
        if true_value is not None:
            ax.axhline(y=true_value, color=self.colors['golden_gate'],
                      linestyle='--', linewidth=2, label='True Value')
        # Formatting
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if log_scale:
            ax.set_yscale('log')
        if len(convergence_data) > 1 or true_value is not None:
            ax.legend()
        # Berkeley branding
        self._add_berkeley_branding(fig)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    def plot_samples(self,
                    samples: np.ndarray,
                    labels: Optional[List[str]] = None,
                    plot_type: str = 'histogram',
                    bins: Union[int, str] = 'auto',
                    density: bool = True,
                    alpha: float = 0.7,
                    title: str = "Sample Distribution",
                    save_path: Optional[str] = None,
                    **kwargs) -> plt.Figure:
        """Plot sample distributions.
        Args:
            samples: Sample data (1D or 2D array)
            labels: Variable labels
            plot_type: Type of plot ('histogram', 'scatter', 'pairs')
            bins: Number of histogram bins
            density: Whether to normalize histograms
            alpha: Transparency level
            title: Plot title
            save_path: Path to save figure
            **kwargs: Additional plot arguments
        Returns:
            Figure object
        """
        samples = np.asarray(samples)
        if samples.ndim == 1:
            return self._plot_1d_samples(samples, title, bins, density, alpha, save_path, **kwargs)
        elif samples.ndim == 2:
            if plot_type == 'pairs':
                return self._plot_pairs(samples, labels, title, save_path, **kwargs)
            elif plot_type == 'scatter' and samples.shape[1] == 2:
                return self._plot_2d_scatter(samples, labels, title, save_path, **kwargs)
            else:
                return self._plot_multi_histogram(samples, labels, title, bins,
                                                density, alpha, save_path, **kwargs)
        else:
            raise ValueError("Samples must be 1D or 2D array")
    def _plot_1d_samples(self, samples: np.ndarray, title: str, bins: Union[int, str],
                        density: bool, alpha: float, save_path: Optional[str],
                        **kwargs) -> plt.Figure:
        """Plot 1D sample histogram."""
        fig, ax = plt.subplots(figsize=self.figsize)
        # Histogram
        n, bins_used, patches = ax.hist(samples, bins=bins, density=density,
                                       alpha=alpha, color=self.colors['berkeley_blue'],
                                       edgecolor=self.colors['berkeley_blue'], linewidth=1)
        # Statistics text box
        stats = compute_statistics(samples)
        stats_text = f"Mean: {stats['mean']:.3f}\\nStd: {stats['std']:.3f}\\nMedian: {stats['median']:.3f}"
        props = dict(boxstyle='round', facecolor=self.colors['bay_fog'], alpha=0.8)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density' if density else 'Count')
        ax.set_title(title)
        self._add_berkeley_branding(fig)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    def _plot_2d_scatter(self, samples: np.ndarray, labels: Optional[List[str]],
                        title: str, save_path: Optional[str], **kwargs) -> plt.Figure:
        """Plot 2D scatter plot."""
        fig, ax = plt.subplots(figsize=self.figsize)
        x, y = samples[:, 0], samples[:, 1]
        # Scatter plot with Berkeley colors
        scatter = ax.scatter(x, y, c=self.colors['berkeley_blue'], alpha=0.6, s=20)
        # Marginal histograms (optional)
        if kwargs.get('marginals', False):
            from matplotlib.patches import Rectangle
            # Create marginal axes
            left, bottom, width, height = ax.get_position().bounds
            # Top histogram
            ax_top = fig.add_axes([left, bottom + height, width, 0.2])
            ax_top.hist(x, bins=30, alpha=0.7, color=self.colors['california_gold'])
            ax_top.set_xticks([])
            # Right histogram
            ax_right = fig.add_axes([left + width, bottom, 0.2, height])
            ax_right.hist(y, bins=30, orientation='horizontal', alpha=0.7,
                         color=self.colors['california_gold'])
            ax_right.set_yticks([])
        # Labels
        xlabel = labels[0] if labels and len(labels) > 0 else 'X1'
        ylabel = labels[1] if labels and len(labels) > 1 else 'X2'
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        self._add_berkeley_branding(fig)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    def _plot_multi_histogram(self, samples: np.ndarray, labels: Optional[List[str]],
                             title: str, bins: Union[int, str], density: bool,
                             alpha: float, save_path: Optional[str], **kwargs) -> plt.Figure:
        """Plot multiple histograms."""
        n_vars = samples.shape[1]
        n_cols = min(3, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        if n_vars == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        color_cycle = [self.colors['berkeley_blue'], self.colors['california_gold'],
                      self.colors['founders_rock']]
        for i in range(n_vars):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            color = color_cycle[i % len(color_cycle)]
            ax.hist(samples[:, i], bins=bins, density=density, alpha=alpha,
                   color=color, edgecolor=color, linewidth=1)
            xlabel = labels[i] if labels and i < len(labels) else f'X{i+1}'
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Density' if density else 'Count')
        # Hide unused subplots
        for i in range(n_vars, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.set_visible(False)
        fig.suptitle(title, fontsize=16)
        self._add_berkeley_branding(fig)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    def _plot_pairs(self, samples: np.ndarray, labels: Optional[List[str]],
                   title: str, save_path: Optional[str], **kwargs) -> plt.Figure:
        """Plot pairwise scatter plots."""
        n_vars = samples.shape[1]
        fig, axes = plt.subplots(n_vars, n_vars, figsize=(2.5*n_vars, 2.5*n_vars))
        for i in range(n_vars):
            for j in range(n_vars):
                ax = axes[i, j]
                if i == j:
                    # Diagonal: histogram
                    ax.hist(samples[:, i], bins=20, alpha=0.7,
                           color=self.colors['berkeley_blue'])
                else:
                    # Off-diagonal: scatter
                    ax.scatter(samples[:, j], samples[:, i],
                             c=self.colors['berkeley_blue'], alpha=0.6, s=10)
                # Labels only on edges
                if i == n_vars - 1:
                    xlabel = labels[j] if labels and j < len(labels) else f'X{j+1}'
                    ax.set_xlabel(xlabel)
                if j == 0:
                    ylabel = labels[i] if labels and i < len(labels) else f'X{i+1}'
                    ax.set_ylabel(ylabel)
        fig.suptitle(title, fontsize=16)
        self._add_berkeley_branding(fig)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    def plot_distributions(self,
                          distributions: Dict[str, Dict[str, Any]],
                          x_range: Optional[Tuple[float, float]] = None,
                          title: str = "Probability Distributions",
                          save_path: Optional[str] = None,
                          **kwargs) -> plt.Figure:
        """Plot probability distributions.
        Args:
            distributions: Dictionary of distribution specifications
            x_range: Range for x-axis
            title: Plot title
            save_path: Path to save figure
            **kwargs: Additional plot arguments
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        color_cycle = [
            self.colors['berkeley_blue'],
            self.colors['california_gold'],
            self.colors['founders_rock'],
            self.colors['medalist'],
            self.colors['lawrence']
        ]
        # Determine x-range if not provided
        if x_range is None:
            x_min, x_max = np.inf, -np.inf
            for dist_name, dist_spec in distributions.items():
                # Rough estimate of range
                if 'loc' in dist_spec and 'scale' in dist_spec:
                    loc, scale = dist_spec['loc'], dist_spec['scale']
                    x_min = min(x_min, loc - 4*scale)
                    x_max = max(x_max, loc + 4*scale)
                else:
                    x_min, x_max = -5, 5  # Default range
            x_range = (x_min, x_max)
        x = np.linspace(x_range[0], x_range[1], 1000)
        for i, (dist_name, dist_spec) in enumerate(distributions.items()):
            color = color_cycle[i % len(color_cycle)]
            try:
                from .constants import get_distribution
                dist = get_distribution(**dist_spec)
                y = dist.pdf(x)
                ax.plot(x, y, label=dist_name, color=color, linewidth=2)
                ax.fill_between(x, y, alpha=0.3, color=color)
            except Exception as e:
                warnings.warn(f"Could not plot distribution {dist_name}: {e}")
        ax.set_xlabel('Value')
        ax.set_ylabel('Probability Density')
        ax.set_title(title)
        ax.legend()
        self._add_berkeley_branding(fig)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    def plot_sensitivity(self,
                        sensitivity_result: SensitivityResult,
                        plot_type: str = 'bar',
                        title: str = "Sensitivity Analysis",
                        save_path: Optional[str] = None,
                        **kwargs) -> plt.Figure:
        """Plot sensitivity analysis results.
        Args:
            sensitivity_result: Sensitivity analysis results
            plot_type: Type of plot ('bar', 'tornado', 'pie')
            title: Plot title
            save_path: Path to save figure
            **kwargs: Additional plot arguments
        Returns:
            Figure object
        """
        if plot_type == 'bar':
            return self._plot_sensitivity_bar(sensitivity_result, title, save_path, **kwargs)
        elif plot_type == 'tornado':
            return self._plot_sensitivity_tornado(sensitivity_result, title, save_path, **kwargs)
        elif plot_type == 'pie':
            return self._plot_sensitivity_pie(sensitivity_result, title, save_path, **kwargs)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
    def _plot_sensitivity_bar(self, result: SensitivityResult, title: str,
                             save_path: Optional[str], **kwargs) -> plt.Figure:
        """Plot sensitivity indices as bar chart."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        x_pos = np.arange(len(result.variable_names))
        # First-order indices
        bars1 = ax1.bar(x_pos, result.first_order, color=self.colors['berkeley_blue'],
                       alpha=0.8, label='First-order')
        # Add confidence intervals if available
        if 'first_order_95%' in result.confidence_intervals:
            ci = result.confidence_intervals['first_order_95%']
            yerr = [result.first_order - ci[0], ci[1] - result.first_order]
            ax1.errorbar(x_pos, result.first_order, yerr=yerr, fmt='none',
                        color='black', capsize=3)
        ax1.set_xlabel('Input Variables')
        ax1.set_ylabel('First-order Index')
        ax1.set_title('First-order Sensitivity Indices')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(result.variable_names, rotation=45)
        ax1.set_ylim(0, 1)
        # Total-order indices
        bars2 = ax2.bar(x_pos, result.total_order, color=self.colors['california_gold'],
                       alpha=0.8, label='Total-order')
        if 'total_order_95%' in result.confidence_intervals:
            ci = result.confidence_intervals['total_order_95%']
            yerr = [result.total_order - ci[0], ci[1] - result.total_order]
            ax2.errorbar(x_pos, result.total_order, yerr=yerr, fmt='none',
                        color='black', capsize=3)
        ax2.set_xlabel('Input Variables')
        ax2.set_ylabel('Total-order Index')
        ax2.set_title('Total-order Sensitivity Indices')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(result.variable_names, rotation=45)
        ax2.set_ylim(0, 1)
        fig.suptitle(title, fontsize=16)
        self._add_berkeley_branding(fig)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    def _plot_sensitivity_tornado(self, result: SensitivityResult, title: str,
                                 save_path: Optional[str], **kwargs) -> plt.Figure:
        """Plot sensitivity indices as tornado chart."""
        fig, ax = plt.subplots(figsize=self.figsize)
        # Sort by total-order indices
        sort_indices = np.argsort(result.total_order)[::-1]
        y_pos = np.arange(len(result.variable_names))
        # Plot bars
        ax.barh(y_pos, result.first_order[sort_indices],
               color=self.colors['berkeley_blue'], alpha=0.8,
               label='First-order', height=0.35)
        ax.barh(y_pos + 0.35, result.total_order[sort_indices],
               color=self.colors['california_gold'], alpha=0.8,
               label='Total-order', height=0.35)
        ax.set_xlabel('Sensitivity Index')
        ax.set_ylabel('Input Variables')
        ax.set_title(title)
        ax.set_yticks(y_pos + 0.175)
        ax.set_yticklabels([result.variable_names[i] for i in sort_indices])
        ax.set_xlim(0, 1)
        ax.legend()
        self._add_berkeley_branding(fig)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    def _plot_sensitivity_pie(self, result: SensitivityResult, title: str,
                             save_path: Optional[str], **kwargs) -> plt.Figure:
        """Plot first-order sensitivity indices as pie chart."""
        fig, ax = plt.subplots(figsize=(8, 8))
        # Use first-order indices for pie chart
        indices = result.first_order
        # Add 'Other' category for remaining variance
        remaining = max(0, 1 - np.sum(indices))
        if remaining > 0.01:  # Only show if significant
            labels = result.variable_names + ['Other']
            sizes = np.append(indices, remaining)
        else:
            labels = result.variable_names
            sizes = indices
        # Berkeley colors
        colors = [self.colors['berkeley_blue'], self.colors['california_gold'],
                 self.colors['founders_rock'], self.colors['medalist']]
        colors = colors * (len(sizes) // len(colors) + 1)
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors[:len(sizes)],
                                         autopct='%1.1f%%', startangle=90)
        ax.set_title(title)
        self._add_berkeley_branding(fig)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    def _add_berkeley_branding(self, fig: plt.Figure):
        """Add Berkeley branding to figure."""
        # Add subtle Berkeley logo or text
        fig.text(0.02, 0.02, 'UC Berkeley SciComp', fontsize=8,
                color=self.colors['berkeley_blue'], alpha=0.7)
# Convenience functions
def plot_convergence(convergence_data: Union[List[float], Dict[str, List[float]]],
                    title: str = "Monte Carlo Convergence",
                    save_path: Optional[str] = None,
                    **kwargs) -> plt.Figure:
    """Convenience function for plotting convergence."""
    visualizer = MonteCarloVisualizer()
    return visualizer.plot_convergence(convergence_data, title=title,
                                     save_path=save_path, **kwargs)
def plot_samples(samples: np.ndarray,
                labels: Optional[List[str]] = None,
                plot_type: str = 'histogram',
                title: str = "Sample Distribution",
                save_path: Optional[str] = None,
                **kwargs) -> plt.Figure:
    """Convenience function for plotting samples."""
    visualizer = MonteCarloVisualizer()
    return visualizer.plot_samples(samples, labels=labels, plot_type=plot_type,
                                 title=title, save_path=save_path, **kwargs)
def plot_distributions(distributions: Dict[str, Dict[str, Any]],
                      title: str = "Probability Distributions",
                      save_path: Optional[str] = None,
                      **kwargs) -> plt.Figure:
    """Convenience function for plotting distributions."""
    visualizer = MonteCarloVisualizer()
    return visualizer.plot_distributions(distributions, title=title,
                                       save_path=save_path, **kwargs)
def plot_sensitivity(sensitivity_result: SensitivityResult,
                    plot_type: str = 'bar',
                    title: str = "Sensitivity Analysis",
                    save_path: Optional[str] = None,
                    **kwargs) -> plt.Figure:
    """Convenience function for plotting sensitivity analysis."""
    visualizer = MonteCarloVisualizer()
    return visualizer.plot_sensitivity(sensitivity_result, plot_type=plot_type,
                                     title=title, save_path=save_path, **kwargs)