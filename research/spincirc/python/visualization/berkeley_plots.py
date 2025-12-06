#!/usr/bin/env python3
"""
Berkeley-Themed Plotting for SpinCirc

Professional publication-quality plotting with UC Berkeley visual identity.
Implements the official Berkeley color palette and styling guidelines.

Author: Meshal Alawein <meshal@berkeley.edu>
Copyright © 2025 Meshal Alawein — All rights reserved.
License: MIT
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class BerkeleyColors:
    """UC Berkeley official color palette and styling"""
    
    # Primary colors
    BERKELEY_BLUE = '#003262'      # Primary blue
    CALIFORNIA_GOLD = '#FDB515'    # Primary gold
    
    # Secondary colors
    BLUE_DARK = '#000133'
    GOLD_DARK = '#FC9F13'
    GREEN_DARK = '#00553A'
    ROSE_DARK = '#77072F'
    PURPLE_DARK = '#431170'
    
    # Neutral colors
    GRAY_DARK = '#666666'
    GRAY_MEDIUM = '#999999'
    GRAY_LIGHT = '#D9D9D9'
    BLACK = '#000000'
    WHITE = '#FFFFFF'
    
    # Color sequences for multi-line plots
    SEQUENCE = [BERKELEY_BLUE, CALIFORNIA_GOLD, GREEN_DARK, ROSE_DARK, 
               PURPLE_DARK, BLUE_DARK, GOLD_DARK]
    
    # Gradients
    BLUE_GRADIENT = ['#E6F2FF', '#CCE5FF', '#99CCFF', '#66B3FF', 
                     '#3399FF', '#0080FF', '#003262']
    GOLD_GRADIENT = ['#FFF9E6', '#FFF3CC', '#FFE699', '#FFD966',
                     '#FFCC33', '#FFB515', '#FDB515']
    
    @classmethod
    def get_colormap(cls, name: str = 'berkeley') -> LinearSegmentedColormap:
        """Get Berkeley-themed colormap"""
        
        if name == 'berkeley':
            colors = [cls.BERKELEY_BLUE, cls.CALIFORNIA_GOLD]
        elif name == 'berkeley_blue':
            colors = cls.BLUE_GRADIENT
        elif name == 'berkeley_gold':
            colors = cls.GOLD_GRADIENT
        elif name == 'berkeley_thermal':
            colors = [cls.BERKELEY_BLUE, cls.WHITE, cls.CALIFORNIA_GOLD]
        else:
            colors = cls.SEQUENCE
        
        return LinearSegmentedColormap.from_list(name, colors)

class BerkeleyPlotter:
    """Professional Berkeley-themed plotting class"""
    
    def __init__(self, style: str = 'publication', dpi: int = 300):
        """
        Initialize Berkeley plotter
        
        Args:
            style: Plot style ('publication', 'presentation', 'poster')
            dpi: Resolution for saved figures
        """
        self.style = style
        self.dpi = dpi
        self.colors = BerkeleyColors()
        
        # Configure matplotlib
        self._setup_matplotlib()
        
        # Style-specific settings
        self.style_configs = {
            'publication': {
                'figure_size': (8, 6),
                'font_size': 12,
                'title_size': 14,
                'label_size': 12,
                'tick_size': 10,
                'legend_size': 10,
                'line_width': 2,
                'marker_size': 8
            },
            'presentation': {
                'figure_size': (12, 9),
                'font_size': 16,
                'title_size': 20,
                'label_size': 16,
                'tick_size': 14,
                'legend_size': 14,
                'line_width': 3,
                'marker_size': 10
            },
            'poster': {
                'figure_size': (16, 12),
                'font_size': 24,
                'title_size': 28,
                'label_size': 24,
                'tick_size': 20,
                'legend_size': 20,
                'line_width': 4,
                'marker_size': 12
            }
        }
        
        self.config = self.style_configs[style]
        
    def _setup_matplotlib(self):
        """Configure matplotlib with Berkeley styling"""
        
        # Try new seaborn syntax first, fall back to legacy
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except OSError:
            plt.style.use('seaborn-whitegrid')
        
        # Set Berkeley color cycle
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=self.colors.SEQUENCE)
        
        # Font settings
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
        plt.rcParams['mathtext.fontset'] = 'stix'
        
        # Figure settings
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['savefig.facecolor'] = 'white'
        plt.rcParams['savefig.edgecolor'] = 'none'
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['savefig.pad_inches'] = 0.1
        
        # Grid settings
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['grid.linewidth'] = 0.5
        
        # Spine settings
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.linewidth'] = 1.5
        
    def create_figure(self, figsize: Optional[Tuple[float, float]] = None,
                     nrows: int = 1, ncols: int = 1, **kwargs) -> Tuple[plt.Figure, Any]:
        """Create Berkeley-styled figure"""
        
        figsize = figsize or self.config['figure_size']
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, 
                                facecolor='white', **kwargs)
        
        # Apply Berkeley styling to axes
        if isinstance(axes, np.ndarray):
            for ax in axes.flat:
                self._style_axis(ax)
        else:
            self._style_axis(axes)
        
        return fig, axes
    
    def _style_axis(self, ax):
        """Apply Berkeley styling to a single axis"""
        
        # Font sizes
        ax.tick_params(labelsize=self.config['tick_size'])
        ax.xaxis.label.set_size(self.config['label_size'])
        ax.yaxis.label.set_size(self.config['label_size'])
        
        # Grid
        ax.grid(True, alpha=0.3, linewidth=0.5)
        
        # Spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        # Tick parameters
        ax.tick_params(direction='in', length=4, width=1.5)
        ax.tick_params(which='minor', direction='in', length=2, width=1)
    
    def plot_iv_characteristic(self, voltage: np.ndarray, current: np.ndarray,
                              labels: Optional[List[str]] = None,
                              title: str = 'I-V Characteristic',
                              save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Plot I-V characteristic curves"""
        
        fig, ax = self.create_figure()
        
        if current.ndim == 1:
            current = current.reshape(-1, 1)
        
        n_curves = current.shape[1]
        
        for i in range(n_curves):
            label = labels[i] if labels and i < len(labels) else f'Curve {i+1}'
            color = self.colors.SEQUENCE[i % len(self.colors.SEQUENCE)]
            
            ax.plot(voltage, current[:, i], linewidth=self.config['line_width'],
                   color=color, label=label, marker='o', 
                   markersize=self.config['marker_size'], markevery=len(voltage)//20)
        
        ax.set_xlabel('Voltage (V)', fontsize=self.config['label_size'])
        ax.set_ylabel('Current (A)', fontsize=self.config['label_size'])
        ax.set_title(title, fontsize=self.config['title_size'], fontweight='bold')
        
        if labels:
            ax.legend(fontsize=self.config['legend_size'], frameon=True, 
                     fancybox=True, shadow=True)
        
        if save_path:
            self.save_figure(fig, save_path)
        
        return fig, ax
    
    def plot_magnetization_trajectory(self, time: np.ndarray, magnetization: np.ndarray,
                                    title: str = 'Magnetization Dynamics',
                                    save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Plot 3D magnetization trajectory"""
        
        fig = plt.figure(figsize=self.config['figure_size'])
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract components
        mx, my, mz = magnetization[0, :], magnetization[1, :], magnetization[2, :]
        
        # Color by time
        colors = plt.cm.viridis(np.linspace(0, 1, len(time)))
        
        # Plot trajectory
        ax.plot(mx, my, mz, linewidth=self.config['line_width'], 
               color=self.colors.BERKELEY_BLUE, alpha=0.8)
        
        # Scatter plot colored by time
        scatter = ax.scatter(mx, my, mz, c=time, cmap='viridis', 
                           s=self.config['marker_size'], alpha=0.6)
        
        # Unit sphere
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')
        
        # Labels and title
        ax.set_xlabel('mx', fontsize=self.config['label_size'])
        ax.set_ylabel('my', fontsize=self.config['label_size'])
        ax.set_zlabel('mz', fontsize=self.config['label_size'])
        ax.set_title(title, fontsize=self.config['title_size'], fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
        cbar.set_label('Time (s)', fontsize=self.config['label_size'])
        
        # Equal aspect ratio
        ax.set_box_aspect([1,1,1])
        
        if save_path:
            self.save_figure(fig, save_path)
        
        return fig, ax
    
    def plot_hysteresis_loop(self, field: np.ndarray, magnetization: np.ndarray,
                           title: str = 'Hysteresis Loop',
                           save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Plot magnetic hysteresis loop"""
        
        fig, ax = self.create_figure()
        
        # Plot loop with gradient coloring
        points = np.array([field, magnetization]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        from matplotlib.collections import LineCollection
        
        colors = np.linspace(0, 1, len(segments))
        lc = LineCollection(segments, cmap='viridis', linewidths=self.config['line_width'])
        lc.set_array(colors)
        
        line = ax.add_collection(lc)
        
        # Add arrows to show direction
        n_arrows = 8
        arrow_indices = np.linspace(0, len(field)-2, n_arrows, dtype=int)
        
        for i in arrow_indices:
            dx = field[i+1] - field[i]
            dy = magnetization[i+1] - magnetization[i]
            ax.annotate('', xy=(field[i+1], magnetization[i+1]),
                       xytext=(field[i], magnetization[i]),
                       arrowprops=dict(arrowstyle='->', color=self.colors.BERKELEY_BLUE,
                                     lw=2, alpha=0.7))
        
        ax.set_xlabel('Applied Field (T)', fontsize=self.config['label_size'])
        ax.set_ylabel('Magnetization (A/m)', fontsize=self.config['label_size'])
        ax.set_title(title, fontsize=self.config['title_size'], fontweight='bold')
        
        # Add coercivity and remanence annotations
        # Find coercive field (zero crossing)
        zero_crossings = np.where(np.diff(np.signbit(magnetization)))[0]
        if len(zero_crossings) >= 2:
            hc1 = np.interp(0, [magnetization[zero_crossings[0]], magnetization[zero_crossings[0]+1]],
                           [field[zero_crossings[0]], field[zero_crossings[0]+1]])
            hc2 = np.interp(0, [magnetization[zero_crossings[1]], magnetization[zero_crossings[1]+1]],
                           [field[zero_crossings[1]], field[zero_crossings[1]+1]])
            hc_avg = abs(hc1 + hc2) / 2
            
            ax.text(0.05, 0.95, f'Hc ≈ {hc_avg:.3f} T', transform=ax.transAxes,
                   fontsize=self.config['tick_size'], bbox=dict(boxstyle='round',
                   facecolor=self.colors.CALIFORNIA_GOLD, alpha=0.8))
        
        plt.colorbar(line, ax=ax, label='Sweep Progress')
        
        if save_path:
            self.save_figure(fig, save_path)
        
        return fig, ax
    
    def plot_transport_sweep(self, sweep_param: np.ndarray, 
                           transport_data: Dict[str, np.ndarray],
                           sweep_name: str = 'Parameter',
                           title: str = 'Transport Characteristics',
                           save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Plot transport parameter sweep"""
        
        n_plots = len(transport_data)
        fig, axes = self.create_figure(nrows=min(n_plots, 2), ncols=(n_plots + 1) // 2)
        
        if n_plots == 1:
            axes = [axes]
        elif isinstance(axes, np.ndarray):
            axes = axes.flatten()
        
        for i, (param_name, values) in enumerate(transport_data.items()):
            ax = axes[i]
            
            color = self.colors.SEQUENCE[i % len(self.colors.SEQUENCE)]
            
            ax.plot(sweep_param, values, linewidth=self.config['line_width'],
                   color=color, marker='o', markersize=self.config['marker_size'],
                   markevery=len(sweep_param)//20)
            
            ax.set_xlabel(sweep_name, fontsize=self.config['label_size'])
            ax.set_ylabel(param_name, fontsize=self.config['label_size'])
            ax.set_title(f'{param_name} vs {sweep_name}', 
                        fontsize=self.config['title_size'])
            
            # Add statistics annotation
            mean_val = np.mean(values)
            std_val = np.std(values)
            ax.text(0.05, 0.95, f'μ = {mean_val:.2e}\nσ = {std_val:.2e}',
                   transform=ax.transAxes, fontsize=self.config['tick_size'],
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=self.config['title_size'], fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            self.save_figure(fig, save_path)
        
        return fig, axes
    
    def plot_device_schematic(self, device_type: str = 'mtj',
                            save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Plot schematic diagram of spintronic device"""
        
        fig, ax = self.create_figure()
        
        if device_type.lower() == 'mtj':
            self._draw_mtj_schematic(ax)
        elif device_type.lower() == 'spin_valve':
            self._draw_spin_valve_schematic(ax)
        elif device_type.lower() == 'asl':
            self._draw_asl_schematic(ax)
        else:
            raise ValueError(f"Unknown device type: {device_type}")
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.set_aspect('equal')
        ax.axis('off')
        
        if save_path:
            self.save_figure(fig, save_path)
        
        return fig, ax
    
    def _draw_mtj_schematic(self, ax):
        """Draw MTJ schematic"""
        
        # Free layer
        free_layer = patches.Rectangle((2, 4), 6, 0.5, 
                                     facecolor=self.colors.BERKELEY_BLUE,
                                     edgecolor='black', linewidth=2)
        ax.add_patch(free_layer)
        ax.text(5, 4.25, 'Free Layer', ha='center', va='center', 
               fontsize=self.config['label_size'], color='white', weight='bold')
        
        # Barrier
        barrier = patches.Rectangle((2, 3.3), 6, 0.2,
                                  facecolor=self.colors.CALIFORNIA_GOLD,
                                  edgecolor='black', linewidth=1)
        ax.add_patch(barrier)
        ax.text(8.5, 3.4, 'MgO', ha='center', va='center',
               fontsize=self.config['tick_size'])
        
        # Reference layer
        ref_layer = patches.Rectangle((2, 2.5), 6, 0.5,
                                    facecolor=self.colors.GREEN_DARK,
                                    edgecolor='black', linewidth=2)
        ax.add_patch(ref_layer)
        ax.text(5, 2.75, 'Reference Layer', ha='center', va='center',
               fontsize=self.config['label_size'], color='white', weight='bold')
        
        # Contacts
        top_contact = patches.Rectangle((4, 4.5), 2, 0.3,
                                      facecolor=self.colors.GRAY_DARK,
                                      edgecolor='black', linewidth=1)
        ax.add_patch(top_contact)
        
        bottom_contact = patches.Rectangle((4, 1.8), 2, 0.3,
                                         facecolor=self.colors.GRAY_DARK,
                                         edgecolor='black', linewidth=1)
        ax.add_patch(bottom_contact)
        
        # Magnetization arrows
        ax.annotate('', xy=(7.5, 4.25), xytext=(6.5, 4.25),
                   arrowprops=dict(arrowstyle='->', color='red', lw=3))
        ax.annotate('', xy=(6.5, 2.75), xytext=(7.5, 2.75),
                   arrowprops=dict(arrowstyle='->', color='red', lw=3))
        
        ax.text(5, 5.5, 'Magnetic Tunnel Junction', ha='center', va='center',
               fontsize=self.config['title_size'], weight='bold')
    
    def _draw_spin_valve_schematic(self, ax):
        """Draw spin valve schematic"""
        
        # Layers from bottom to top
        layers = [
            ('Substrate', 0.5, self.colors.GRAY_LIGHT),
            ('Bottom FM', 1.0, self.colors.BERKELEY_BLUE), 
            ('Spacer', 1.5, self.colors.CALIFORNIA_GOLD),
            ('Top FM', 2.0, self.colors.GREEN_DARK),
            ('Cap', 2.5, self.colors.GRAY_DARK)
        ]
        
        for name, y_pos, color in layers:
            layer = patches.Rectangle((2, y_pos), 6, 0.4,
                                    facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(layer)
            ax.text(8.5, y_pos + 0.2, name, ha='left', va='center',
                   fontsize=self.config['tick_size'])
        
        # Current path
        ax.annotate('', xy=(1.5, 2.7), xytext=(1.5, 0.7),
                   arrowprops=dict(arrowstyle='->', color='blue', lw=4))
        ax.text(1, 1.7, 'I', ha='center', va='center',
               fontsize=self.config['label_size'], color='blue', weight='bold')
        
        ax.text(5, 3.5, 'Giant Magnetoresistance Spin Valve', ha='center', va='center',
               fontsize=self.config['title_size'], weight='bold')
    
    def _draw_asl_schematic(self, ax):
        """Draw All-Spin Logic schematic"""
        
        # Channel
        channel = patches.Rectangle((2, 2.5), 6, 1,
                                  facecolor=self.colors.CALIFORNIA_GOLD,
                                  edgecolor='black', linewidth=2, alpha=0.7)
        ax.add_patch(channel)
        ax.text(5, 3, 'Spin Channel', ha='center', va='center',
               fontsize=self.config['label_size'], weight='bold')
        
        # Input magnet
        input_mag = patches.Ellipse((1.5, 3), 0.8, 0.6,
                                  facecolor=self.colors.BERKELEY_BLUE,
                                  edgecolor='black', linewidth=2)
        ax.add_patch(input_mag)
        ax.text(1.5, 3, 'Input', ha='center', va='center',
               fontsize=self.config['tick_size'], color='white', weight='bold')
        
        # Output magnet
        output_mag = patches.Ellipse((8.5, 3), 0.8, 0.6,
                                   facecolor=self.colors.GREEN_DARK,
                                   edgecolor='black', linewidth=2)
        ax.add_patch(output_mag)
        ax.text(8.5, 3, 'Output', ha='center', va='center',
               fontsize=self.config['tick_size'], color='white', weight='bold')
        
        # Spin current arrow
        ax.annotate('', xy=(7.8, 3), xytext=(2.2, 3),
                   arrowprops=dict(arrowstyle='->', color='red', lw=4))
        ax.text(5, 3.7, 'Spin Current', ha='center', va='center',
               fontsize=self.config['tick_size'], color='red', weight='bold')
        
        ax.text(5, 4.5, 'All-Spin Logic Device', ha='center', va='center',
               fontsize=self.config['title_size'], weight='bold')
    
    def plot_parameter_sensitivity(self, sensitivity_data: Dict[str, float],
                                 method: str = 'sobol',
                                 title: str = 'Parameter Sensitivity Analysis',
                                 save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Plot parameter sensitivity analysis results"""
        
        fig, ax = self.create_figure(figsize=(10, 6))
        
        params = list(sensitivity_data.keys())
        values = list(sensitivity_data.values())
        
        # Sort by sensitivity
        sorted_indices = np.argsort(values)[::-1]
        params_sorted = [params[i] for i in sorted_indices]
        values_sorted = [values[i] for i in sorted_indices]
        
        # Create horizontal bar plot
        bars = ax.barh(params_sorted, values_sorted, 
                      color=self.colors.BERKELEY_BLUE, alpha=0.8)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, values_sorted)):
            ax.text(value + max(values_sorted)*0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}', ha='left', va='center',
                   fontsize=self.config['tick_size'])
        
        ax.set_xlabel(f'{method.capitalize()} Sensitivity Index',
                     fontsize=self.config['label_size'])
        ax.set_title(title, fontsize=self.config['title_size'], fontweight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='x')
        
        if save_path:
            self.save_figure(fig, save_path)
        
        return fig, ax
    
    def plot_uncertainty_distribution(self, data: np.ndarray,
                                    parameter_name: str = 'Output',
                                    title: str = 'Uncertainty Distribution',
                                    save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Plot uncertainty distribution with confidence intervals"""
        
        fig, (ax1, ax2) = self.create_figure(nrows=1, ncols=2, figsize=(12, 5))
        
        # Histogram with KDE
        ax1.hist(data, bins=50, density=True, alpha=0.7, 
                color=self.colors.BERKELEY_BLUE, edgecolor='black')
        
        # Kernel density estimate
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 200)
        ax1.plot(x_range, kde(x_range), color=self.colors.CALIFORNIA_GOLD,
                linewidth=self.config['line_width'], label='KDE')
        
        # Statistics
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        ax1.axvline(mean_val, color='red', linestyle='--', 
                   linewidth=2, label=f'Mean = {mean_val:.3f}')
        ax1.axvline(mean_val - std_val, color='red', linestyle=':', 
                   linewidth=1.5, alpha=0.7)
        ax1.axvline(mean_val + std_val, color='red', linestyle=':', 
                   linewidth=1.5, alpha=0.7)
        
        ax1.set_xlabel(parameter_name, fontsize=self.config['label_size'])
        ax1.set_ylabel('Density', fontsize=self.config['label_size'])
        ax1.set_title('Distribution', fontsize=self.config['title_size'])
        ax1.legend(fontsize=self.config['legend_size'])
        
        # Box plot
        bp = ax2.boxplot(data, patch_artist=True, labels=[parameter_name])
        bp['boxes'][0].set_facecolor(self.colors.BERKELEY_BLUE)
        bp['boxes'][0].set_alpha(0.7)
        
        ax2.set_ylabel(parameter_name, fontsize=self.config['label_size'])
        ax2.set_title('Box Plot', fontsize=self.config['title_size'])
        
        # Add statistics text
        percentiles = np.percentile(data, [5, 25, 50, 75, 95])
        stats_text = f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}\n'
        stats_text += f'P5: {percentiles[0]:.3f}\nP95: {percentiles[4]:.3f}'
        
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                fontsize=self.config['tick_size'], va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle(title, fontsize=self.config['title_size'], fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            self.save_figure(fig, save_path)
        
        return fig, (ax1, ax2)
    
    def save_figure(self, fig: plt.Figure, filepath: str, **kwargs):
        """Save figure with Berkeley standards"""
        
        # Default save parameters
        save_params = {
            'dpi': self.dpi,
            'bbox_inches': 'tight',
            'facecolor': 'white',
            'edgecolor': 'none',
            'pad_inches': 0.1
        }
        
        save_params.update(kwargs)
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        fig.savefig(filepath, **save_params)
        logger.info(f"Figure saved to {filepath}")
    
    def create_color_palette_demo(self, save_path: Optional[str] = None):
        """Create demonstration of Berkeley color palette"""
        
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # Primary colors
        axes[0].bar(['Berkeley Blue', 'California Gold'], [1, 1],
                   color=[self.colors.BERKELEY_BLUE, self.colors.CALIFORNIA_GOLD])
        axes[0].set_title('Primary Colors')
        
        # Secondary colors
        secondary_colors = [self.colors.BLUE_DARK, self.colors.GOLD_DARK,
                          self.colors.GREEN_DARK, self.colors.ROSE_DARK,
                          self.colors.PURPLE_DARK]
        secondary_names = ['Blue Dark', 'Gold Dark', 'Green Dark', 'Rose Dark', 'Purple Dark']
        
        axes[1].bar(range(len(secondary_colors)), [1]*len(secondary_colors),
                   color=secondary_colors)
        axes[1].set_xticks(range(len(secondary_names)))
        axes[1].set_xticklabels(secondary_names, rotation=45)
        axes[1].set_title('Secondary Colors')
        
        # Color sequence
        x = np.linspace(0, 10, 100)
        for i, color in enumerate(self.colors.SEQUENCE[:5]):
            axes[2].plot(x, np.sin(x + i), color=color, linewidth=3, 
                        label=f'Color {i+1}')
        axes[2].set_title('Color Sequence')
        axes[2].legend()
        
        # Gradients
        gradient_data = np.random.randn(10, 10)
        
        im1 = axes[3].imshow(gradient_data, cmap=self.colors.get_colormap('berkeley_blue'))
        axes[3].set_title('Blue Gradient')
        plt.colorbar(im1, ax=axes[3])
        
        im2 = axes[4].imshow(gradient_data, cmap=self.colors.get_colormap('berkeley_gold'))
        axes[4].set_title('Gold Gradient')
        plt.colorbar(im2, ax=axes[4])
        
        im3 = axes[5].imshow(gradient_data, cmap=self.colors.get_colormap('berkeley_thermal'))
        axes[5].set_title('Thermal Gradient')
        plt.colorbar(im3, ax=axes[5])
        
        plt.suptitle('UC Berkeley Color Palette for SpinCirc', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            self.save_figure(fig, save_path)
        
        return fig, axes