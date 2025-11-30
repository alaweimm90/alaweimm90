"""
UC Berkeley themed plotting styles and color palettes.

This module provides the official UC Berkeley visual identity for
scientific plots and visualizations, ensuring consistency across
all MagLogic publications and presentations.

Author: Meshal Alawein
Email: meshal@berkeley.edu
License: MIT
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rcParams
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import warnings

# UC Berkeley Official Color Palette
BERKELEY_COLORS = {
    "primary": {
        "berkeley_blue": "#003262",
        "california_gold": "#FDB515"
    },
    "secondary": {
        "blue_dark": "#010133", 
        "gold_dark": "#FC9313",
        "green_dark": "#00553A",
        "rose_dark": "#770747", 
        "purple_dark": "#431170",
        "red_dark": "#8C1515",
        "orange_dark": "#D2691E",
        "teal_dark": "#004C5A"
    },
    "neutral": {
        "grey_light": "#D9D9D9",
        "grey_medium": "#999999",
        "grey_dark": "#666666", 
        "black": "#000000",
        "white": "#FFFFFF"
    },
    "accent": {
        "sky_blue": "#87CEEB",
        "mint_green": "#98FB98", 
        "coral": "#FF7F50",
        "lavender": "#E6E6FA",
        "peach": "#FFCBA4"
    }
}

class BerkeleyStyle:
    """
    UC Berkeley themed plotting style manager.
    
    This class provides comprehensive styling for matplotlib plots
    following UC Berkeley's official visual identity guidelines.
    
    Attributes:
        colors: Dictionary of UC Berkeley colors
        figure_params: Default figure parameters
        font_params: Typography settings
        
    Example:
        >>> style = BerkeleyStyle()
        >>> style.setup()  # Apply Berkeley styling globally
        >>> plt.plot([1, 2, 3], [1, 4, 2])
        >>> plt.show()  # Plot with Berkeley styling
    """
    
    def __init__(self):
        """Initialize Berkeley style with default parameters."""
        self.colors = BERKELEY_COLORS
        self._original_rcparams = rcParams.copy()
        self._style_applied = False
        
        # Default figure parameters
        self.figure_params = {
            "figure.figsize": (8, 6),
            "figure.dpi": 100,
            "figure.facecolor": self.colors["neutral"]["white"],
            "figure.edgecolor": self.colors["neutral"]["white"],
            "savefig.dpi": 300,
            "savefig.facecolor": self.colors["neutral"]["white"],
            "savefig.edgecolor": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        }
        
        # Typography parameters
        self.font_params = {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.titleweight": "bold",
            "axes.labelweight": "normal",
        }
        
        # Axes and layout parameters
        self.axes_params = {
            "axes.facecolor": self.colors["neutral"]["white"],
            "axes.edgecolor": self.colors["neutral"]["black"],
            "axes.linewidth": 1.5,
            "axes.titlecolor": self.colors["primary"]["berkeley_blue"],
            "axes.labelcolor": self.colors["neutral"]["black"],
            "axes.prop_cycle": plt.cycler("color", self.get_color_cycle()),
            "axes.grid": False,
            "axes.axisbelow": True,
        }
        
        # Grid parameters
        self.grid_params = {
            "grid.color": self.colors["neutral"]["grey_light"],
            "grid.linestyle": "-",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.7,
        }
        
        # Tick parameters
        self.tick_params = {
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
            "xtick.major.width": 1,
            "ytick.major.width": 1,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "xtick.color": self.colors["neutral"]["black"],
            "ytick.color": self.colors["neutral"]["black"],
        }
        
        # Line and marker parameters
        self.line_params = {
            "lines.linewidth": 2,
            "lines.markersize": 8,
            "lines.markeredgewidth": 1,
            "lines.solid_capstyle": "round",
            "patch.linewidth": 1,
            "patch.edgecolor": self.colors["neutral"]["black"],
        }
        
        # Legend parameters
        self.legend_params = {
            "legend.frameon": True,
            "legend.fancybox": False,
            "legend.shadow": False,
            "legend.framealpha": 1.0,
            "legend.facecolor": self.colors["neutral"]["white"],
            "legend.edgecolor": self.colors["neutral"]["grey_dark"],
            "legend.borderpad": 0.4,
            "legend.columnspacing": 2.0,
            "legend.handlelength": 2.0,
            "legend.handletextpad": 0.8,
        }
    
    def get_color_cycle(self) -> List[str]:
        """
        Get Berkeley-themed color cycle for plots.
        
        Returns:
            List of color hex codes in preferred order
        """
        return [
            self.colors["primary"]["berkeley_blue"],
            self.colors["primary"]["california_gold"], 
            self.colors["secondary"]["green_dark"],
            self.colors["secondary"]["red_dark"],
            self.colors["secondary"]["purple_dark"],
            self.colors["secondary"]["orange_dark"],
            self.colors["secondary"]["teal_dark"],
            self.colors["secondary"]["rose_dark"]
        ]
    
    def get_colormap(self, name: str = "berkeley") -> mcolors.LinearSegmentedColormap:
        """
        Generate Berkeley-themed colormaps.
        
        Args:
            name: Colormap name ('berkeley', 'berkeley_blue', 'berkeley_gold',
                 'magnetization', 'energy')
                 
        Returns:
            Custom colormap
            
        Raises:
            ValueError: If colormap name is unknown
        """
        if name == "berkeley":
            colors = [
                self.colors["primary"]["berkeley_blue"],
                self.colors["neutral"]["white"], 
                self.colors["primary"]["california_gold"]
            ]
        elif name == "berkeley_blue":
            colors = [
                self.colors["neutral"]["white"],
                self.colors["primary"]["berkeley_blue"]
            ]
        elif name == "berkeley_gold":
            colors = [
                self.colors["neutral"]["white"],
                self.colors["primary"]["california_gold"]
            ]
        elif name == "magnetization":
            colors = [
                self.colors["secondary"]["red_dark"],    # -1 (down)
                self.colors["neutral"]["white"],         #  0 (neutral)
                self.colors["primary"]["berkeley_blue"]  # +1 (up)
            ]
        elif name == "energy":
            colors = [
                self.colors["primary"]["berkeley_blue"], 
                self.colors["accent"]["sky_blue"],
                self.colors["neutral"]["white"],
                self.colors["accent"]["peach"],
                self.colors["primary"]["california_gold"]
            ]
        elif name == "phase":
            # HSV-like colormap for phase/angle data
            colors = [
                self.colors["secondary"]["red_dark"],
                self.colors["secondary"]["orange_dark"],
                self.colors["primary"]["california_gold"],
                self.colors["secondary"]["green_dark"],
                self.colors["primary"]["berkeley_blue"],
                self.colors["secondary"]["purple_dark"],
                self.colors["secondary"]["rose_dark"]
            ]
        else:
            raise ValueError(f"Unknown colormap: {name}")
        
        return mcolors.LinearSegmentedColormap.from_list(name, colors)
    
    def setup(self) -> None:
        """Apply Berkeley styling to matplotlib globally."""
        if self._style_applied:
            return
        
        # Combine all parameter dictionaries
        all_params = {}
        for param_dict in [
            self.figure_params,
            self.font_params, 
            self.axes_params,
            self.grid_params,
            self.tick_params,
            self.line_params,
            self.legend_params
        ]:
            all_params.update(param_dict)
        
        # Apply parameters
        plt.rcParams.update(all_params)
        
        # Register custom colormaps
        self._register_colormaps()
        
        self._style_applied = True
    
    def reset(self) -> None:
        """Reset matplotlib to original settings."""
        if self._style_applied:
            plt.rcParams.clear()
            plt.rcParams.update(self._original_rcparams)
            self._style_applied = False
    
    def _register_colormaps(self) -> None:
        """Register custom colormaps with matplotlib."""
        colormap_names = [
            "berkeley", "berkeley_blue", "berkeley_gold", 
            "magnetization", "energy", "phase"
        ]
        
        for name in colormap_names:
            try:
                cmap = self.get_colormap(name)
                plt.cm.register_cmap(name, cmap)
            except Exception as e:
                warnings.warn(f"Could not register colormap '{name}': {e}")
    
    def apply_to_figure(self, fig: plt.Figure, title: Optional[str] = None) -> None:
        """
        Apply Berkeley styling to an existing figure.
        
        Args:
            fig: Matplotlib figure object
            title: Optional super title for the figure
        """
        # Set figure properties
        fig.patch.set_facecolor(self.colors["neutral"]["white"])
        fig.patch.set_edgecolor(self.colors["neutral"]["white"]) 
        
        # Apply super title if provided
        if title:
            fig.suptitle(
                title,
                fontsize=self.font_params["axes.titlesize"],
                fontweight=self.font_params["axes.titleweight"],
                color=self.colors["primary"]["berkeley_blue"]
            )
        
        # Apply to all axes
        for ax in fig.get_axes():
            self.apply_to_axes(ax)
    
    def apply_to_axes(self, ax: plt.Axes) -> None:
        """
        Apply Berkeley styling to matplotlib axes.
        
        Args:
            ax: Matplotlib axes object
        """
        # Set face and edge colors
        ax.set_facecolor(self.colors["neutral"]["white"])
        
        # Style spines
        for spine in ax.spines.values():
            spine.set_color(self.colors["neutral"]["black"])
            spine.set_linewidth(self.axes_params["axes.linewidth"])
        
        # Style ticks
        ax.tick_params(
            colors=self.colors["neutral"]["black"],
            direction=self.tick_params["xtick.direction"],
            length=self.tick_params["xtick.major.size"],
            width=self.tick_params["xtick.major.width"]
        )
        
        # Style labels
        ax.xaxis.label.set_color(self.colors["neutral"]["black"])
        ax.yaxis.label.set_color(self.colors["neutral"]["black"])
        
        # Style title
        if ax.get_title():
            ax.title.set_color(self.colors["primary"]["berkeley_blue"])
            ax.title.set_fontweight(self.font_params["axes.titleweight"])
    
    def create_publication_figure(self, 
                                figsize: Tuple[float, float] = (8, 6),
                                nrows: int = 1, 
                                ncols: int = 1,
                                **kwargs) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
        """
        Create a publication-ready figure with Berkeley styling.
        
        Args:
            figsize: Figure size in inches
            nrows: Number of subplot rows
            ncols: Number of subplot columns
            **kwargs: Additional arguments for plt.subplots()
            
        Returns:
            Figure and axes objects
        """
        # Ensure styling is applied
        if not self._style_applied:
            self.setup()
        
        # Create figure
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
        
        # Apply styling
        self.apply_to_figure(fig)
        
        return fig, axes
    
    def save_figure(self, 
                   fig: plt.Figure,
                   filename: Union[str, Path],
                   dpi: int = 300,
                   format: str = "png",
                   **kwargs) -> None:
        """
        Save figure with Berkeley styling and high quality.
        
        Args:
            fig: Figure to save
            filename: Output filename
            dpi: Resolution in dots per inch
            format: File format ('png', 'pdf', 'svg', 'eps')
            **kwargs: Additional arguments for fig.savefig()
        """
        # Default save parameters
        save_params = {
            "dpi": dpi,
            "format": format,
            "facecolor": self.colors["neutral"]["white"],
            "edgecolor": "none",
            "bbox_inches": "tight",
            "pad_inches": 0.1,
        }
        save_params.update(kwargs)
        
        # Save figure
        fig.savefig(filename, **save_params)
    
    def create_color_palette(self, n_colors: int = 8) -> List[str]:
        """
        Create a color palette with specified number of colors.
        
        Args:
            n_colors: Number of colors to generate
            
        Returns:
            List of color hex codes
        """
        base_colors = self.get_color_cycle()
        
        if n_colors <= len(base_colors):
            return base_colors[:n_colors]
        else:
            # Generate additional colors by interpolating
            extended_colors = base_colors.copy()
            
            while len(extended_colors) < n_colors:
                # Add lighter/darker variants
                for base_color in base_colors:
                    if len(extended_colors) >= n_colors:
                        break
                    
                    # Create lighter variant
                    rgb = mcolors.hex2color(base_color)
                    lighter = tuple(min(1.0, c + 0.3) for c in rgb)
                    extended_colors.append(mcolors.rgb2hex(lighter))
                    
                    if len(extended_colors) >= n_colors:
                        break
                    
                    # Create darker variant  
                    darker = tuple(max(0.0, c - 0.3) for c in rgb)
                    extended_colors.append(mcolors.rgb2hex(darker))
            
            return extended_colors[:n_colors]
    
    def get_color(self, name: str) -> str:
        """
        Get a specific color by name.
        
        Args:
            name: Color name (e.g., 'berkeley_blue', 'california_gold')
            
        Returns:
            Color hex code
            
        Raises:
            KeyError: If color name is not found
        """
        # Search through all color categories
        for category in self.colors.values():
            if isinstance(category, dict) and name in category:
                return category[name]
        
        # If not found, raise error with available colors
        available_colors = []
        for category in self.colors.values():
            if isinstance(category, dict):
                available_colors.extend(category.keys())
        
        raise KeyError(f"Color '{name}' not found. Available colors: {available_colors}")
    
    def print_colors(self) -> None:
        """Print all available colors with their hex codes."""
        print("UC Berkeley Color Palette")
        print("=" * 40)
        
        for category, colors in self.colors.items():
            print(f"\n{category.upper()}:")
            for name, hex_code in colors.items():
                print(f"  {name}: {hex_code}")

# Create global instance
berkeley_style = BerkeleyStyle()

def create_berkeley_colormap(name: str = "berkeley") -> mcolors.LinearSegmentedColormap:
    """
    Create and register Berkeley colormaps with matplotlib.
    
    Args:
        name: Colormap name
        
    Returns:
        Custom colormap
    """
    return berkeley_style.get_colormap(name)

def setup_berkeley_style() -> None:
    """Set up Berkeley styling globally."""
    berkeley_style.setup()

def reset_style() -> None:
    """Reset matplotlib to original settings."""
    berkeley_style.reset()

def berkeley_colors() -> Dict:
    """Get the complete Berkeley color dictionary."""
    return berkeley_style.colors

def berkeley_palette(n_colors: int = 8) -> List[str]:
    """
    Get Berkeley color palette.
    
    Args:
        n_colors: Number of colors
        
    Returns:
        List of color hex codes
    """
    return berkeley_style.create_color_palette(n_colors)

# Specialized colormaps for micromagnetic data
def magnetization_colormap() -> mcolors.LinearSegmentedColormap:
    """Colormap specifically for magnetization data (Mz component)."""
    return berkeley_style.get_colormap("magnetization")

def energy_colormap() -> mcolors.LinearSegmentedColormap:
    """Colormap for energy density visualization.""" 
    return berkeley_style.get_colormap("energy")

def phase_colormap() -> mcolors.LinearSegmentedColormap:
    """Colormap for phase/angle data visualization."""
    return berkeley_style.get_colormap("phase")

# Context manager for temporary styling
class BerkeleyStyleContext:
    """Context manager for temporary Berkeley styling."""
    
    def __init__(self):
        self.original_applied = berkeley_style._style_applied
    
    def __enter__(self):
        berkeley_style.setup()
        return berkeley_style
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.original_applied:
            berkeley_style.reset()

def with_berkeley_style():
    """Decorator for applying Berkeley style to functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with BerkeleyStyleContext():
                return func(*args, **kwargs)
        return wrapper
    return decorator

# Initialize style on import (optional)
try:
    berkeley_style.setup()
except Exception as e:
    warnings.warn(f"Could not initialize Berkeley style: {e}")

# Export color constants for direct use
BERKELEY_BLUE = BERKELEY_COLORS["primary"]["berkeley_blue"]
CALIFORNIA_GOLD = BERKELEY_COLORS["primary"]["california_gold"]
BERKELEY_PALETTE = berkeley_style.get_color_cycle()