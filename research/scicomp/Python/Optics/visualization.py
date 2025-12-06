"""Visualization Module for Optics.
This module provides Berkeley-themed visualization tools for optical systems,
beam profiles, ray diagrams, and interference patterns.
Author: Berkeley SciComp Team
Date: 2024
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
class OpticsVisualizer:
    """Berkeley-themed optical visualization class."""
    def __init__(self):
        """Initialize visualizer with Berkeley styling."""
        # Berkeley color scheme
        self.berkeley_blue = '#003262'
        self.california_gold = '#FDB515'
        self.berkeley_light_blue = '#3B7EA1'
        self.berkeley_dark_gold = '#C4820E'
        self.berkeley_secondary_blue = '#00B0DA'
        # Color palette
        self.colors = [
            self.berkeley_blue,
            self.california_gold,
            self.berkeley_light_blue,
            self.berkeley_dark_gold,
            self.berkeley_secondary_blue
        ]
        # Apply Berkeley styling
        self.setup_berkeley_style()
    def setup_berkeley_style(self):
        """Setup Berkeley visual styling for matplotlib."""
        plt.rcParams.update({
            'figure.figsize': [10, 6],
            'axes.prop_cycle': plt.cycler('color', self.colors),
            'axes.grid': True,
            'grid.alpha': 0.3,
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'legend.fontsize': 10,
            'lines.linewidth': 2,
            'axes.linewidth': 1.2,
            'xtick.major.size': 6,
            'ytick.major.size': 6,
            'xtick.minor.size': 4,
            'ytick.minor.size': 4
        })
    def plot_beam_profile(self, x: np.ndarray, y: np.ndarray, z: float,
                         intensity: np.ndarray, title: str = "Beam Profile",
                         figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """Plot 2D beam intensity profile.
        Args:
            x, y: Coordinate arrays (meters)
            z: Axial position (meters)
            intensity: 2D intensity array (W/m²)
            title: Plot title
            figsize: Figure size
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        # 2D intensity map
        X, Y = np.meshgrid(x, y)
        im = ax1.contourf(X * 1000, Y * 1000, intensity, levels=50, cmap='Blues')
        ax1.set_xlabel('X Position (mm)')
        ax1.set_ylabel('Y Position (mm)')
        ax1.set_title(f'{title} at z = {z*1000:.1f} mm')
        ax1.set_aspect('equal')
        # Colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Intensity (W/m²)', fontsize=12)
        # Cross-sections
        center_idx = len(y) // 2
        x_profile = intensity[center_idx, :]
        y_profile = intensity[:, len(x) // 2]
        ax2.plot(x * 1000, x_profile, color=self.berkeley_blue,
                linewidth=2, label='Horizontal')
        ax2.plot(y * 1000, y_profile, color=self.california_gold,
                linewidth=2, label='Vertical')
        ax2.set_xlabel('Position (mm)')
        ax2.set_ylabel('Intensity (W/m²)')
        ax2.set_title('Beam Cross-sections')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    def plot_ray_diagram(self, rays: List, surfaces: List = None,
                        title: str = "Ray Diagram",
                        figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Plot ray diagram with optical elements.
        Args:
            rays: List of ray objects or ray trace results
            surfaces: List of optical surfaces
            title: Plot title
            figsize: Figure size
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        # Plot rays
        for i, ray_data in enumerate(rays):
            if hasattr(ray_data, 'rays'):  # Ray trace result
                ray_list = ray_data.rays
            else:  # Direct ray list
                ray_list = ray_data
            # Extract ray path
            z_positions = []
            x_positions = []
            for ray in ray_list:
                z_positions.append(ray.position[2])
                x_positions.append(ray.position[0])
            # Plot ray path
            color = self.colors[i % len(self.colors)]
            ax.plot(np.array(z_positions) * 1000, np.array(x_positions) * 1000,
                   color=color, linewidth=1.5, alpha=0.8)
        # Plot optical surfaces
        if surfaces:
            for surface in surfaces:
                self._draw_optical_element(ax, surface)
        ax.set_xlabel('Optical Axis (mm)')
        ax.set_ylabel('Height (mm)')
        ax.set_title(title, fontsize=14, color=self.berkeley_blue, fontweight='bold')
        ax.grid(True, alpha=0.3)
        # Equal aspect ratio for proper visualization
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        return fig
    def _draw_optical_element(self, ax: plt.Axes, surface) -> None:
        """Draw optical element on ray diagram."""
        if hasattr(surface, 'position'):
            z_pos = surface.position * 1000  # Convert to mm
            if hasattr(surface, 'focal_length'):  # Thin lens
                # Draw lens symbol
                height = getattr(surface, 'diameter', 20) * 1000 / 2
                ax.axvline(z_pos, ymin=0.5 - height/100, ymax=0.5 + height/100,
                          color=self.california_gold, linewidth=4, alpha=0.8)
                # Add focal points
                if surface.focal_length > 0:
                    f_pos = z_pos + surface.focal_length * 1000
                    ax.plot(f_pos, 0, 'o', color=self.berkeley_blue, markersize=6)
            elif hasattr(surface, 'radius'):  # Curved surface
                # Draw curved surface
                if abs(surface.radius) > 1e12:  # Flat surface
                    ax.axvline(z_pos, color=self.berkeley_blue, linewidth=2)
                else:
                    # Simplified curved surface representation
                    height = getattr(surface, 'diameter', 20) * 1000 / 2
                    curve_x = np.linspace(-height/10, height/10, 100)
                    curve_y = np.linspace(-height, height, 100)
                    ax.plot(z_pos + curve_x, curve_y, color=self.berkeley_blue, linewidth=2)
    def plot_interference_pattern(self, x: np.ndarray, intensity: np.ndarray,
                                 pattern_info: Dict = None,
                                 title: str = "Interference Pattern",
                                 figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """Plot interference pattern.
        Args:
            x: Position array (meters)
            intensity: Intensity array
            pattern_info: Additional pattern information
            title: Plot title
            figsize: Figure size
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        # Plot intensity pattern
        ax.plot(x * 1000, intensity, color=self.berkeley_blue, linewidth=2)
        ax.fill_between(x * 1000, 0, intensity, alpha=0.3, color=self.berkeley_blue)
        ax.set_xlabel('Position (mm)')
        ax.set_ylabel('Intensity (arbitrary units)')
        ax.set_title(title, fontsize=14, color=self.berkeley_blue, fontweight='bold')
        ax.grid(True, alpha=0.3)
        # Add pattern information if provided
        if pattern_info:
            info_text = []
            if 'wavelength' in pattern_info:
                info_text.append(f"λ = {pattern_info['wavelength']*1e9:.1f} nm")
            if 'fringe_spacing' in pattern_info:
                info_text.append(f"Δx = {pattern_info['fringe_spacing']*1000:.3f} mm")
            if 'visibility' in pattern_info:
                info_text.append(f"V = {pattern_info['visibility']:.3f}")
            if info_text:
                ax.text(0.02, 0.98, '\n'.join(info_text),
                       transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=dict(boxstyle='round',
                       facecolor='white', alpha=0.8))
        plt.tight_layout()
        return fig
    def plot_diffraction_pattern(self, x: np.ndarray, intensity: np.ndarray,
                                aperture_info: Dict = None,
                                title: str = "Diffraction Pattern",
                                figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """Plot diffraction pattern with aperture information.
        Args:
            x: Position array (meters)
            intensity: Intensity array
            aperture_info: Aperture parameters
            title: Plot title
            figsize: Figure size
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
        # Main diffraction pattern
        ax1.plot(x * 1000, intensity, color=self.berkeley_blue, linewidth=2)
        ax1.fill_between(x * 1000, 0, intensity, alpha=0.3, color=self.berkeley_blue)
        ax1.set_ylabel('Intensity')
        ax1.set_title(title, fontsize=14, color=self.berkeley_blue, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        # Log scale plot
        ax2.semilogy(x * 1000, intensity, color=self.california_gold, linewidth=2)
        ax2.set_xlabel('Position (mm)')
        ax2.set_ylabel('Intensity (log)')
        ax2.grid(True, alpha=0.3)
        # Add aperture information
        if aperture_info:
            info_text = []
            if 'aperture_type' in aperture_info:
                info_text.append(f"Aperture: {aperture_info['aperture_type']}")
            if 'aperture_size' in aperture_info:
                size_mm = aperture_info['aperture_size'] * 1000
                info_text.append(f"Size: {size_mm:.3f} mm")
            if 'wavelength' in aperture_info:
                wl_nm = aperture_info['wavelength'] * 1e9
                info_text.append(f"λ = {wl_nm:.1f} nm")
            if info_text:
                ax1.text(0.02, 0.98, '\n'.join(info_text),
                        transform=ax1.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round',
                        facecolor='white', alpha=0.8))
        plt.tight_layout()
        return fig
    def animate_wave_propagation(self, x: np.ndarray, z: np.ndarray,
                               field_function: callable, wavelength: float,
                               title: str = "Wave Propagation",
                               save_path: str = None) -> FuncAnimation:
        """Create animation of wave propagation.
        Args:
            x: Spatial coordinates (meters)
            z: Propagation distances (meters)
            field_function: Function(x, z, t) returning field amplitude
            wavelength: Wavelength (meters)
            title: Animation title
            save_path: Path to save animation (optional)
        Returns:
            Animation object
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        # Initialize plot
        line_real, = ax.plot([], [], color=self.berkeley_blue,
                           linewidth=2, label='Real part')
        line_imag, = ax.plot([], [], color=self.california_gold,
                           linewidth=2, label='Imaginary part')
        line_intensity, = ax.plot([], [], color=self.berkeley_light_blue,
                                linewidth=2, label='Intensity')
        ax.set_xlim(x[0] * 1000, x[-1] * 1000)
        ax.set_ylim(-2, 2)
        ax.set_xlabel('Position (mm)')
        ax.set_ylabel('Amplitude')
        ax.set_title(title, fontsize=14, color=self.berkeley_blue, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        # Time parameters
        period = wavelength / 2.99792458e8  # seconds
        num_frames = 60
        time_points = np.linspace(0, 2*period, num_frames)
        def animate(frame):
            t = time_points[frame]
            z_current = z[min(frame * len(z) // num_frames, len(z) - 1)]
            # Calculate field at current time and position
            field = field_function(x, z_current, t)
            intensity = np.abs(field)**2
            # Update lines
            line_real.set_data(x * 1000, np.real(field))
            line_imag.set_data(x * 1000, np.imag(field))
            line_intensity.set_data(x * 1000, intensity)
            # Update title with current parameters
            ax.set_title(f'{title} - z = {z_current*1000:.1f} mm, t = {t*1e15:.1f} fs',
                        fontsize=14, color=self.berkeley_blue, fontweight='bold')
            return line_real, line_imag, line_intensity
        # Create animation
        anim = FuncAnimation(fig, animate, frames=num_frames, interval=100, blit=True)
        if save_path:
            anim.save(save_path, writer='pillow', fps=10)
        plt.tight_layout()
        return anim
    def create_optical_system_diagram(self, elements: List[Dict],
                                    title: str = "Optical System",
                                    figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
        """Create comprehensive optical system diagram.
        Args:
            elements: List of optical element dictionaries
            title: Diagram title
            figsize: Figure size
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        # Draw optical axis
        z_min = min(elem.get('position', 0) for elem in elements) - 50e-3
        z_max = max(elem.get('position', 0) for elem in elements) + 50e-3
        ax.axhline(0, color='black', linewidth=1, alpha=0.5, linestyle='--')
        # Draw elements
        for i, element in enumerate(elements):
            self._draw_system_element(ax, element, i)
        # Formatting
        ax.set_xlim(z_min * 1000, z_max * 1000)
        ax.set_xlabel('Optical Axis (mm)')
        ax.set_ylabel('Height (mm)')
        ax.set_title(title, fontsize=16, color=self.berkeley_blue, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        return fig
    def _draw_system_element(self, ax: plt.Axes, element: Dict, index: int) -> None:
        """Draw individual optical element in system diagram."""
        elem_type = element.get('type', 'unknown')
        position = element.get('position', 0) * 1000  # Convert to mm
        color = self.colors[index % len(self.colors)]
        if elem_type == 'lens':
            # Draw lens
            focal_length = element.get('focal_length', 0.1)
            diameter = element.get('diameter', 25e-3) * 1000 / 2
            # Lens symbol
            ax.plot([position, position], [-diameter, diameter],
                   color=color, linewidth=4, alpha=0.8)
            # Focal points
            if focal_length > 0:
                f_pos = position + focal_length * 1000
                ax.plot(f_pos, 0, 'o', color=color, markersize=6)
                ax.plot(position - focal_length * 1000, 0, 'o',
                       color=color, markersize=6, fillstyle='none')
            # Label
            ax.text(position, diameter + 5, f'L{index+1}',
                   ha='center', va='bottom', color=color, fontweight='bold')
        elif elem_type == 'mirror':
            # Draw mirror
            diameter = element.get('diameter', 25e-3) * 1000 / 2
            ax.plot([position, position], [-diameter, diameter],
                   color=color, linewidth=6, alpha=0.8)
            # Label
            ax.text(position, diameter + 5, f'M{index+1}',
                   ha='center', va='bottom', color=color, fontweight='bold')
        elif elem_type == 'aperture':
            # Draw aperture
            diameter = element.get('diameter', 10e-3) * 1000 / 2
            ax.plot([position, position], [-50, -diameter],
                   color=color, linewidth=3)
            ax.plot([position, position], [diameter, 50],
                   color=color, linewidth=3)
            # Label
            ax.text(position, 30, f'A{index+1}',
                   ha='center', va='bottom', color=color, fontweight='bold')
# Convenience functions
def plot_beam_profile(x: np.ndarray, y: np.ndarray, z: float,
                     intensity: np.ndarray, **kwargs) -> plt.Figure:
    """Plot beam profile using Berkeley styling."""
    viz = OpticsVisualizer()
    return viz.plot_beam_profile(x, y, z, intensity, **kwargs)
def plot_ray_diagram(rays: List, surfaces: List = None, **kwargs) -> plt.Figure:
    """Plot ray diagram using Berkeley styling."""
    viz = OpticsVisualizer()
    return viz.plot_ray_diagram(rays, surfaces, **kwargs)
def plot_interference_pattern(x: np.ndarray, intensity: np.ndarray,
                            pattern_info: Dict = None, **kwargs) -> plt.Figure:
    """Plot interference pattern using Berkeley styling."""
    viz = OpticsVisualizer()
    return viz.plot_interference_pattern(x, intensity, pattern_info, **kwargs)
def animate_wave_propagation(x: np.ndarray, z: np.ndarray,
                           field_function: callable, wavelength: float,
                           **kwargs) -> FuncAnimation:
    """Create wave propagation animation using Berkeley styling."""
    viz = OpticsVisualizer()
    return viz.animate_wave_propagation(x, z, field_function, wavelength, **kwargs)
def create_optical_system_diagram(elements: List[Dict], **kwargs) -> plt.Figure:
    """Create optical system diagram using Berkeley styling."""
    viz = OpticsVisualizer()
    return viz.create_optical_system_diagram(elements, **kwargs)
# Demo function
def demo_visualization():
    """Demonstrate visualization capabilities."""
    print("Optics Visualization Demo")
    print("========================")
    viz = OpticsVisualizer()
    # 1. Gaussian beam profile
    print("\n1. Gaussian Beam Profile")
    x = np.linspace(-5e-3, 5e-3, 100)
    y = np.linspace(-5e-3, 5e-3, 100)
    X, Y = np.meshgrid(x, y)
    # Gaussian intensity profile
    w0 = 1e-3  # Beam waist
    intensity = np.exp(-2 * (X**2 + Y**2) / w0**2)
    fig1 = viz.plot_beam_profile(x, y, 0, intensity, title="Gaussian Beam")
    plt.show()
    # 2. Interference pattern
    print("\n2. Interference Pattern")
    x_screen = np.linspace(-5e-3, 5e-3, 1000)
    wavelength = 633e-9
    slit_separation = 100e-6
    # Double slit interference
    phase_diff = 2 * np.pi * slit_separation * x_screen / (wavelength * 1.0)
    intensity_interference = 4 * np.cos(phase_diff / 2)**2
    pattern_info = {
        'wavelength': wavelength,
        'fringe_spacing': wavelength * 1.0 / slit_separation,
        'visibility': 1.0
    }
    fig2 = viz.plot_interference_pattern(x_screen, intensity_interference,
                                       pattern_info, title="Double Slit Interference")
    plt.show()
    # 3. Optical system diagram
    print("\n3. Optical System Diagram")
    elements = [
        {'type': 'lens', 'position': 0, 'focal_length': 0.1, 'diameter': 25e-3},
        {'type': 'aperture', 'position': 0.05, 'diameter': 5e-3},
        {'type': 'lens', 'position': 0.15, 'focal_length': 0.05, 'diameter': 20e-3}
    ]
    fig3 = viz.create_optical_system_diagram(elements, title="Telescope System")
    plt.show()
    print("\nVisualization demo completed!")
if __name__ == "__main__":
    demo_visualization()