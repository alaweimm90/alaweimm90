"""Berkeley-themed Visualization for Multiphysics Simulations.
This module provides comprehensive visualization utilities for multiphysics
simulations with UC Berkeley's official branding and color scheme.
Classes:
    MultiphysicsVisualizer: Main visualization class
Functions:
    plot_coupled_fields: Plot multiple physics fields
    plot_interface_data: Plot interface coupling data
    animate_multiphysics: Create animations
    create_multiphysics_plot: Generate publication plots
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
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import Union, List, Dict, Optional, Tuple, Any
import warnings
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
# Multiphysics-specific colors
PHYSICS_COLORS = {
    'thermal': BERKELEY_COLORS['golden_gate'],
    'mechanical': BERKELEY_COLORS['berkeley_blue'],
    'electromagnetic': BERKELEY_COLORS['lawrence'],
    'fluid': BERKELEY_COLORS['lap_lane'],
    'structure': BERKELEY_COLORS['founders_rock'],
    'transport': BERKELEY_COLORS['ion'],
    'acoustic': BERKELEY_COLORS['rose_garden'],
    'chemical': BERKELEY_COLORS['wellman_tile']
}
# Plotting parameters
BERKELEY_STYLE = {
    'figure.figsize': [12, 8],
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
class MultiphysicsVisualizer:
    """Comprehensive multiphysics visualization toolkit.
    Provides publication-ready plots for multiphysics analysis results
    with consistent Berkeley branding.
    """
    def __init__(self, style: str = 'berkeley', figsize: Tuple[float, float] = (12, 8)):
        self.style = style
        self.figsize = figsize
        self.colors = BERKELEY_COLORS.copy()
        self.physics_colors = PHYSICS_COLORS.copy()
        # Apply Berkeley style
        if style == 'berkeley':
            plt.style.use('default')  # Reset first
            plt.rcParams.update(BERKELEY_STYLE)
    def plot_coupled_fields(self,
                           mesh: Dict[str, np.ndarray],
                           field_data: Dict[str, np.ndarray],
                           field_names: Optional[List[str]] = None,
                           title: str = "Coupled Multiphysics Fields",
                           save_path: Optional[str] = None,
                           **kwargs) -> plt.Figure:
        """Plot multiple coupled physics fields.
        Args:
            mesh: Mesh data with 'nodes' and 'elements'
            field_data: Dictionary of field arrays
            field_names: Names for each field
            title: Plot title
            save_path: Path to save figure
            **kwargs: Additional plot arguments
        Returns:
            Figure object
        """
        n_fields = len(field_data)
        if n_fields == 0:
            raise ValueError("No field data provided")
        # Determine subplot layout
        if n_fields <= 2:
            fig, axes = plt.subplots(1, n_fields, figsize=(6*n_fields, 6))
        elif n_fields <= 4:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        elif n_fields <= 6:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        else:
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        if n_fields == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        # Plot each field
        for i, (field_name, field_values) in enumerate(field_data.items()):
            if i >= len(axes):
                break
            ax = axes[i]
            # Get physics type for coloring
            physics_type = self._identify_physics_type(field_name)
            base_color = self.physics_colors.get(physics_type, self.colors['berkeley_blue'])
            # 2D contour plot
            if mesh['nodes'].shape[1] >= 2:
                x = mesh['nodes'][:, 0]
                y = mesh['nodes'][:, 1]
                # Create contour plot
                if len(field_values) == len(x):
                    # Scalar field
                    scatter = ax.scatter(x, y, c=field_values, cmap='viridis', s=20)
                    plt.colorbar(scatter, ax=ax, shrink=0.8)
                else:
                    # Vector field or other
                    ax.scatter(x, y, c=base_color, s=20, alpha=0.6)
            else:
                # 1D plot
                ax.plot(mesh['nodes'][:, 0], field_values,
                       color=base_color, linewidth=2)
            # Formatting
            display_name = field_names[i] if field_names and i < len(field_names) else field_name
            ax.set_title(f"{display_name}", fontsize=12, color=self.colors['berkeley_blue'])
            ax.set_xlabel('X')
            if mesh['nodes'].shape[1] >= 2:
                ax.set_ylabel('Y')
            ax.grid(True, alpha=0.3)
        # Hide unused subplots
        for i in range(n_fields, len(axes)):
            axes[i].set_visible(False)
        fig.suptitle(title, fontsize=16, color=self.colors['berkeley_blue'])
        self._add_berkeley_branding(fig)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    def plot_interface_data(self,
                           interface_nodes: np.ndarray,
                           interface_data: Dict[str, np.ndarray],
                           interface_name: str = "FSI Interface",
                           title: str = "Interface Coupling Data",
                           save_path: Optional[str] = None,
                           **kwargs) -> plt.Figure:
        """Plot data at coupling interfaces.
        Args:
            interface_nodes: Interface node coordinates
            interface_data: Data at interface (forces, displacements, etc.)
            interface_name: Name of interface
            title: Plot title
            save_path: Path to save figure
            **kwargs: Additional plot arguments
        Returns:
            Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # Left plot: Interface geometry
        ax1 = axes[0]
        ax1.plot(interface_nodes[:, 0], interface_nodes[:, 1],
                'o-', color=self.colors['berkeley_blue'], linewidth=2, markersize=6)
        ax1.set_title(f"{interface_name} Geometry")
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        # Right plot: Interface data
        ax2 = axes[1]
        # Plot interface quantities
        if len(interface_data) > 0:
            data_names = list(interface_data.keys())
            if len(data_names) == 1:
                # Single quantity
                data_name = data_names[0]
                data_values = interface_data[data_name]
                if data_values.ndim == 1:
                    # Scalar data along interface
                    s = np.linspace(0, 1, len(data_values))  # Normalized arc length
                    ax2.plot(s, data_values, 'o-', color=self.colors['california_gold'],
                            linewidth=2, markersize=6)
                    ax2.set_xlabel('Normalized Arc Length')
                    ax2.set_ylabel(data_name)
                else:
                    # Vector data - plot magnitude
                    magnitude = np.linalg.norm(data_values, axis=1)
                    s = np.linspace(0, 1, len(magnitude))
                    ax2.plot(s, magnitude, 'o-', color=self.colors['california_gold'],
                            linewidth=2, markersize=6)
                    ax2.set_xlabel('Normalized Arc Length')
                    ax2.set_ylabel(f'|{data_name}|')
            else:
                # Multiple quantities
                colors = [self.colors['california_gold'], self.colors['founders_rock'],
                         self.colors['lawrence'], self.colors['rose_garden']]
                for i, (data_name, data_values) in enumerate(interface_data.items()):
                    if i >= len(colors):
                        break
                    if data_values.ndim == 1:
                        s = np.linspace(0, 1, len(data_values))
                        ax2.plot(s, data_values, 'o-', color=colors[i],
                                linewidth=2, markersize=4, label=data_name)
                    else:
                        magnitude = np.linalg.norm(data_values, axis=1)
                        s = np.linspace(0, 1, len(magnitude))
                        ax2.plot(s, magnitude, 'o-', color=colors[i],
                                linewidth=2, markersize=4, label=f'|{data_name}|')
                ax2.legend()
                ax2.set_xlabel('Normalized Arc Length')
                ax2.set_ylabel('Interface Quantities')
        ax2.set_title("Interface Data")
        ax2.grid(True, alpha=0.3)
        fig.suptitle(title, fontsize=16, color=self.colors['berkeley_blue'])
        self._add_berkeley_branding(fig)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    def plot_convergence_history(self,
                                convergence_data: Dict[str, List[float]],
                                title: str = "Multiphysics Convergence",
                                log_scale: bool = True,
                                save_path: Optional[str] = None,
                                **kwargs) -> plt.Figure:
        """Plot convergence history for coupled solvers.
        Args:
            convergence_data: Dictionary of convergence histories
            title: Plot title
            log_scale: Use logarithmic y-axis
            save_path: Path to save figure
            **kwargs: Additional plot arguments
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        colors = [self.colors['berkeley_blue'], self.colors['california_gold'],
                 self.colors['founders_rock'], self.colors['lawrence']]
        for i, (solver_name, history) in enumerate(convergence_data.items()):
            color = colors[i % len(colors)]
            iterations = range(1, len(history) + 1)
            if log_scale:
                ax.semilogy(iterations, history, 'o-', color=color,
                           linewidth=2, markersize=6, label=solver_name)
            else:
                ax.plot(iterations, history, 'o-', color=color,
                       linewidth=2, markersize=6, label=solver_name)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Residual' if log_scale else 'Residual')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        # Add tolerance line if specified
        tolerance = kwargs.get('tolerance', None)
        if tolerance is not None:
            ax.axhline(y=tolerance, color=self.colors['golden_gate'],
                      linestyle='--', linewidth=2, label='Tolerance')
            ax.legend()
        self._add_berkeley_branding(fig)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    def animate_multiphysics(self,
                           mesh: Dict[str, np.ndarray],
                           time_history: Dict[str, List[np.ndarray]],
                           field_name: str,
                           title: str = "Multiphysics Animation",
                           save_path: Optional[str] = None,
                           fps: int = 10,
                           **kwargs) -> FuncAnimation:
        """Create animation of multiphysics evolution.
        Args:
            mesh: Mesh data
            time_history: Time history of fields
            field_name: Field to animate
            title: Animation title
            save_path: Path to save animation
            fps: Frames per second
            **kwargs: Additional animation arguments
        Returns:
            Animation object
        """
        if field_name not in time_history:
            raise ValueError(f"Field {field_name} not found in time history")
        field_history = time_history[field_name]
        time_values = kwargs.get('time_values', range(len(field_history)))
        # Setup figure
        fig, ax = plt.subplots(figsize=self.figsize)
        # Get data ranges
        all_values = np.concatenate(field_history)
        vmin, vmax = np.min(all_values), np.max(all_values)
        # Initial plot
        x = mesh['nodes'][:, 0]
        y = mesh['nodes'][:, 1]
        scatter = ax.scatter(x, y, c=field_history[0],
                           vmin=vmin, vmax=vmax, cmap='viridis', s=20)
        colorbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        colorbar.set_label(field_name)
        # Animation function
        def animate(frame):
            scatter.set_array(field_history[frame])
            ax.set_title(f"{title} - t = {time_values[frame]:.3f}")
            return [scatter]
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(field_history),
                           interval=1000//fps, blit=False, repeat=True)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
        self._add_berkeley_branding(fig)
        if save_path:
            anim.save(save_path, writer='pillow', fps=fps, dpi=150)
        return anim
    def plot_3d_multiphysics(self,
                           mesh_3d: Dict[str, np.ndarray],
                           field_data: np.ndarray,
                           field_name: str = "Field",
                           title: str = "3D Multiphysics Visualization",
                           save_path: Optional[str] = None,
                           **kwargs) -> plt.Figure:
        """Create 3D visualization of multiphysics fields.
        Args:
            mesh_3d: 3D mesh data
            field_data: 3D field values
            field_name: Name of field
            title: Plot title
            save_path: Path to save figure
            **kwargs: Additional plot arguments
        Returns:
            Figure object
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        x = mesh_3d['nodes'][:, 0]
        y = mesh_3d['nodes'][:, 1]
        z = mesh_3d['nodes'][:, 2]
        # 3D scatter plot with color mapping
        scatter = ax.scatter(x, y, z, c=field_data, cmap='viridis', s=20)
        # Colorbar
        colorbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
        colorbar.set_label(field_name)
        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        # Berkeley branding
        self._add_berkeley_branding(fig)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    def create_coupling_diagram(self,
                               physics_domains: List[str],
                               coupling_interfaces: List[Tuple[str, str]],
                               title: str = "Multiphysics Coupling Diagram",
                               save_path: Optional[str] = None,
                               **kwargs) -> plt.Figure:
        """Create coupling diagram showing physics interactions.
        Args:
            physics_domains: List of physics domain names
            coupling_interfaces: List of (source, target) coupling pairs
            title: Diagram title
            save_path: Path to save figure
            **kwargs: Additional arguments
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        # Arrange physics domains in circle
        n_domains = len(physics_domains)
        angles = np.linspace(0, 2*np.pi, n_domains, endpoint=False)
        radius = 3.0
        positions = {}
        for i, domain in enumerate(physics_domains):
            x = radius * np.cos(angles[i])
            y = radius * np.sin(angles[i])
            positions[domain] = (x, y)
            # Draw domain circle
            physics_type = self._identify_physics_type(domain)
            color = self.physics_colors.get(physics_type, self.colors['berkeley_blue'])
            circle = plt.Circle((x, y), 0.5, color=color, alpha=0.7)
            ax.add_patch(circle)
            # Domain label
            ax.text(x, y, domain, ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white')
        # Draw coupling arrows
        for source, target in coupling_interfaces:
            if source in positions and target in positions:
                x1, y1 = positions[source]
                x2, y2 = positions[target]
                # Arrow from source to target
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                           arrowprops=dict(arrowstyle='->', lw=2,
                                         color=self.colors['california_gold']))
        # Formatting
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=16, color=self.colors['berkeley_blue'])
        ax.axis('off')
        self._add_berkeley_branding(fig)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    def _identify_physics_type(self, field_name: str) -> str:
        """Identify physics type from field name."""
        field_lower = field_name.lower()
        if any(term in field_lower for term in ['temp', 'heat', 'thermal']):
            return 'thermal'
        elif any(term in field_lower for term in ['disp', 'stress', 'strain', 'mech']):
            return 'mechanical'
        elif any(term in field_lower for term in ['elect', 'magnet', 'current']):
            return 'electromagnetic'
        elif any(term in field_lower for term in ['vel', 'press', 'fluid', 'flow']):
            return 'fluid'
        elif any(term in field_lower for term in ['conc', 'species', 'transport']):
            return 'transport'
        elif any(term in field_lower for term in ['struct', 'solid']):
            return 'structure'
        else:
            return 'other'
    def _add_berkeley_branding(self, fig: plt.Figure):
        """Add Berkeley branding to figure."""
        fig.text(0.02, 0.02, 'UC Berkeley SciComp - Multiphysics',
                fontsize=8, color=self.colors['berkeley_blue'], alpha=0.7)
# Convenience functions
def plot_coupled_fields(mesh: Dict[str, np.ndarray],
                        field_data: Dict[str, np.ndarray],
                        title: str = "Coupled Fields",
                        save_path: Optional[str] = None,
                        **kwargs) -> plt.Figure:
    """Convenience function for plotting coupled fields."""
    visualizer = MultiphysicsVisualizer()
    return visualizer.plot_coupled_fields(mesh, field_data, title=title,
                                        save_path=save_path, **kwargs)
def plot_interface_data(interface_nodes: np.ndarray,
                       interface_data: Dict[str, np.ndarray],
                       title: str = "Interface Data",
                       save_path: Optional[str] = None,
                       **kwargs) -> plt.Figure:
    """Convenience function for plotting interface data."""
    visualizer = MultiphysicsVisualizer()
    return visualizer.plot_interface_data(interface_nodes, interface_data,
                                        title=title, save_path=save_path, **kwargs)
def animate_multiphysics(mesh: Dict[str, np.ndarray],
                        time_history: Dict[str, List[np.ndarray]],
                        field_name: str,
                        title: str = "Multiphysics Animation",
                        save_path: Optional[str] = None,
                        **kwargs) -> FuncAnimation:
    """Convenience function for creating multiphysics animations."""
    visualizer = MultiphysicsVisualizer()
    return visualizer.animate_multiphysics(mesh, time_history, field_name,
                                         title=title, save_path=save_path, **kwargs)
def create_multiphysics_plot(solution_data: Dict[str, Any],
                           plot_type: str = "fields",
                           title: str = "Multiphysics Results",
                           save_path: Optional[str] = None,
                           **kwargs) -> plt.Figure:
    """Create comprehensive multiphysics plot.
    Args:
        solution_data: Multiphysics solution data
        plot_type: Type of plot (fields, convergence, coupling)
        title: Plot title
        save_path: Path to save figure
        **kwargs: Additional arguments
    Returns:
        Figure object
    """
    visualizer = MultiphysicsVisualizer()
    if plot_type == "fields":
        mesh = solution_data.get('mesh', {})
        fields = solution_data.get('fields', {})
        return visualizer.plot_coupled_fields(mesh, fields, title=title,
                                            save_path=save_path, **kwargs)
    elif plot_type == "convergence":
        convergence = solution_data.get('convergence_history', {})
        return visualizer.plot_convergence_history(convergence, title=title,
                                                  save_path=save_path, **kwargs)
    elif plot_type == "coupling":
        domains = solution_data.get('physics_domains', [])
        interfaces = solution_data.get('coupling_interfaces', [])
        return visualizer.create_coupling_diagram(domains, interfaces, title=title,
                                                 save_path=save_path, **kwargs)
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")
def plot_physics_comparison(results_dict: Dict[str, Dict[str, np.ndarray]],
                          mesh: Dict[str, np.ndarray],
                          field_name: str,
                          title: str = "Physics Comparison",
                          save_path: Optional[str] = None,
                          **kwargs) -> plt.Figure:
    """Compare results from different physics or methods.
    Args:
        results_dict: Dictionary of {method_name: field_data}
        mesh: Mesh data
        field_name: Field to compare
        title: Plot title
        save_path: Path to save figure
        **kwargs: Additional arguments
    Returns:
        Figure object
    """
    n_methods = len(results_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 5))
    if n_methods == 1:
        axes = [axes]
    for i, (method_name, field_data) in enumerate(results_dict.items()):
        ax = axes[i]
        if field_name in field_data:
            field_values = field_data[field_name]
            x = mesh['nodes'][:, 0]
            y = mesh['nodes'][:, 1]
            scatter = ax.scatter(x, y, c=field_values, cmap='viridis', s=20)
            plt.colorbar(scatter, ax=ax, shrink=0.8)
        ax.set_title(f"{method_name}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig