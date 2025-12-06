"""Visualization for ODE and PDE Solutions.
This module provides comprehensive visualization capabilities for ODE and PDE
solutions with UC Berkeley's official branding and scientific plotting standards.
Classes:
    ODEPDEVisualizer: Main visualization class for ODE/PDE results
Functions:
    plot_ode_solution: Plot ODE solution trajectories
    plot_pde_solution: Plot PDE solutions (1D, 2D, 3D)
    animate_pde_evolution: Create animations of time-dependent PDEs
    create_phase_portrait: Create phase portraits for ODE systems
    plot_convergence_study: Visualize numerical convergence
    plot_stability_region: Plot stability regions for methods
Author: Berkeley SciComp Team
Date: 2024
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import Union, List, Dict, Optional, Tuple, Any, Callable
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
# Method-specific colors
METHOD_COLORS = {
    'euler': BERKELEY_COLORS['golden_gate'],
    'rk4': BERKELEY_COLORS['berkeley_blue'],
    'rk45': BERKELEY_COLORS['founders_rock'],
    'adams': BERKELEY_COLORS['lawrence'],
    'bdf': BERKELEY_COLORS['lap_lane'],
    'finite_difference': BERKELEY_COLORS['berkeley_blue'],
    'finite_element': BERKELEY_COLORS['california_gold'],
    'spectral': BERKELEY_COLORS['rose_garden']
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
class ODEPDEVisualizer:
    """Comprehensive ODE and PDE visualization toolkit.
    Provides publication-ready plots for differential equation solutions
    with consistent Berkeley branding.
    """
    def __init__(self, style: str = 'berkeley', figsize: Tuple[float, float] = (10, 6)):
        """Initialize visualizer.
        Args:
            style: Plotting style ('berkeley', 'default')
            figsize: Figure size
        """
        self.style = style
        self.figsize = figsize
        self.colors = BERKELEY_COLORS.copy()
        self.method_colors = METHOD_COLORS.copy()
        # Apply Berkeley style
        if style == 'berkeley':
            plt.style.use('default')  # Reset first
            plt.rcParams.update(BERKELEY_STYLE)
    def plot_ode_solution(self,
                         t: np.ndarray,
                         y: np.ndarray,
                         labels: Optional[List[str]] = None,
                         title: str = "ODE Solution",
                         xlabel: str = "Time",
                         ylabel: str = "Solution",
                         save_path: Optional[str] = None,
                         **kwargs) -> plt.Figure:
        """Plot ODE solution trajectories.
        Args:
            t: Time points
            y: Solution values (can be multi-dimensional)
            labels: Variable labels
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Path to save figure
            **kwargs: Additional plot arguments
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        # Handle different solution shapes
        if y.ndim == 1:
            # Single variable
            ax.plot(t, y, color=self.colors['berkeley_blue'], linewidth=2,
                   label=labels[0] if labels else None)
        else:
            # Multiple variables
            n_vars = y.shape[0] if y.ndim == 2 else 1
            color_cycle = [self.colors['berkeley_blue'], self.colors['california_gold'],
                          self.colors['founders_rock'], self.colors['lawrence'],
                          self.colors['rose_garden']]
            for i in range(min(n_vars, len(color_cycle))):
                if y.ndim == 2:
                    y_i = y[i, :]
                else:
                    y_i = y
                label = labels[i] if labels and i < len(labels) else f'y_{i+1}'
                ax.plot(t, y_i, color=color_cycle[i], linewidth=2, label=label)
        # Formatting
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, color=self.colors['berkeley_blue'])
        ax.grid(True, alpha=0.3)
        if labels or y.ndim > 1:
            ax.legend()
        self._add_berkeley_branding(fig)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    def plot_pde_solution_1d(self,
                           x: np.ndarray,
                           u: np.ndarray,
                           t: Optional[np.ndarray] = None,
                           title: str = "PDE Solution",
                           xlabel: str = "Space (x)",
                           ylabel: str = "Solution (u)",
                           time_snapshots: Optional[List[int]] = None,
                           save_path: Optional[str] = None,
                           **kwargs) -> plt.Figure:
        """Plot 1D PDE solution.
        Args:
            x: Spatial coordinates
            u: Solution values (time × space or just space)
            t: Time points (for time-dependent)
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            time_snapshots: Specific time indices to plot
            save_path: Path to save figure
            **kwargs: Additional plot arguments
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        if u.ndim == 1:
            # Steady-state solution
            ax.plot(x, u, color=self.colors['berkeley_blue'], linewidth=2,
                   label='u(x)')
        else:
            # Time-dependent solution
            if time_snapshots is None:
                # Plot a few snapshots
                nt = u.shape[0]
                time_snapshots = [0, nt//4, nt//2, 3*nt//4, nt-1]
            colors = [self.colors['berkeley_blue'], self.colors['california_gold'],
                     self.colors['founders_rock'], self.colors['lawrence'],
                     self.colors['rose_garden']]
            for i, t_idx in enumerate(time_snapshots):
                if t_idx < u.shape[0]:
                    color = colors[i % len(colors)]
                    time_label = f't = {t[t_idx]:.3f}' if t is not None else f'Step {t_idx}'
                    ax.plot(x, u[t_idx, :], color=color, linewidth=2,
                           label=time_label)
        # Formatting
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, color=self.colors['berkeley_blue'])
        ax.grid(True, alpha=0.3)
        ax.legend()
        self._add_berkeley_branding(fig)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    def plot_pde_solution_2d(self,
                           x: np.ndarray,
                           y: np.ndarray,
                           u: np.ndarray,
                           title: str = "2D PDE Solution",
                           xlabel: str = "x",
                           ylabel: str = "y",
                           colorbar_label: str = "u",
                           contour_levels: int = 20,
                           save_path: Optional[str] = None,
                           **kwargs) -> plt.Figure:
        """Plot 2D PDE solution as contour plot.
        Args:
            x: X coordinates (1D array)
            y: Y coordinates (1D array)
            u: Solution values (2D array)
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            colorbar_label: Colorbar label
            contour_levels: Number of contour levels
            save_path: Path to save figure
            **kwargs: Additional plot arguments
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        # Create meshgrid if needed
        if x.ndim == 1 and y.ndim == 1:
            X, Y = np.meshgrid(x, y)
        else:
            X, Y = x, y
        # Contour plot
        if kwargs.get('filled', True):
            cs = ax.contourf(X, Y, u, levels=contour_levels, cmap='viridis')
        else:
            cs = ax.contour(X, Y, u, levels=contour_levels, colors=self.colors['berkeley_blue'])
        # Colorbar
        cbar = plt.colorbar(cs, ax=ax, shrink=0.8)
        cbar.set_label(colorbar_label)
        # Formatting
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, color=self.colors['berkeley_blue'])
        ax.set_aspect('equal', adjustable='box')
        self._add_berkeley_branding(fig)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    def animate_pde_evolution(self,
                            x: np.ndarray,
                            u: np.ndarray,
                            t: np.ndarray,
                            title: str = "PDE Evolution",
                            xlabel: str = "Space (x)",
                            ylabel: str = "Solution (u)",
                            save_path: Optional[str] = None,
                            fps: int = 10,
                            **kwargs) -> FuncAnimation:
        """Create animation of PDE time evolution.
        Args:
            x: Spatial coordinates
            u: Solution values (time × space)
            t: Time points
            title: Animation title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Path to save animation
            fps: Frames per second
            **kwargs: Additional animation arguments
        Returns:
            Animation object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        # Set up plot limits
        u_min, u_max = np.min(u), np.max(u)
        margin = 0.1 * (u_max - u_min)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(u_min - margin, u_max + margin)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        # Initialize line
        line, = ax.plot([], [], color=self.colors['berkeley_blue'], linewidth=2)
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                           color=self.colors['berkeley_blue'])
        def animate(frame):
            line.set_data(x, u[frame, :])
            time_text.set_text(f'Time = {t[frame]:.3f}')
            ax.set_title(f"{title} - Frame {frame + 1}/{len(t)}",
                        color=self.colors['berkeley_blue'])
            return line, time_text
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(t),
                           interval=1000//fps, blit=False, repeat=True)
        self._add_berkeley_branding(fig)
        if save_path:
            anim.save(save_path, writer='pillow', fps=fps, dpi=150)
        return anim
    def create_phase_portrait(self,
                            dydt_func: Callable,
                            x_range: Tuple[float, float] = (-3, 3),
                            y_range: Tuple[float, float] = (-3, 3),
                            grid_density: int = 20,
                            trajectories: Optional[List[Tuple[float, float]]] = None,
                            title: str = "Phase Portrait",
                            xlabel: str = "y₁",
                            ylabel: str = "y₂",
                            save_path: Optional[str] = None,
                            **kwargs) -> plt.Figure:
        """Create phase portrait for 2D ODE system.
        Args:
            dydt_func: Function defining dy/dt = f(t, y)
            x_range: Range for x-axis (y₁)
            y_range: Range for y₂-axis (y₂)
            grid_density: Number of grid points per axis
            trajectories: List of initial conditions for trajectories
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Path to save figure
            **kwargs: Additional plot arguments
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        # Create meshgrid for vector field
        y1 = np.linspace(x_range[0], x_range[1], grid_density)
        y2 = np.linspace(y_range[0], y_range[1], grid_density)
        Y1, Y2 = np.meshgrid(y1, y2)
        # Compute vector field
        DY1 = np.zeros_like(Y1)
        DY2 = np.zeros_like(Y2)
        for i in range(grid_density):
            for j in range(grid_density):
                y_vec = np.array([Y1[i, j], Y2[i, j]])
                dydt = dydt_func(0, y_vec)  # Autonomous system
                DY1[i, j] = dydt[0]
                DY2[i, j] = dydt[1]
        # Normalize vectors for better visualization
        M = np.sqrt(DY1**2 + DY2**2)
        M[M == 0] = 1  # Avoid division by zero
        DY1_norm = DY1 / M
        DY2_norm = DY2 / M
        # Plot vector field
        ax.quiver(Y1, Y2, DY1_norm, DY2_norm,
                 M, cmap='Blues', alpha=0.6, scale=30)
        # Plot trajectories if provided
        if trajectories:
            from scipy.integrate import solve_ivp
            colors = [self.colors['california_gold'], self.colors['rose_garden'],
                     self.colors['lawrence'], self.colors['golden_gate']]
            for i, (y1_0, y2_0) in enumerate(trajectories):
                color = colors[i % len(colors)]
                # Integrate trajectory
                sol = solve_ivp(dydt_func, [0, 10], [y1_0, y2_0],
                               dense_output=True, rtol=1e-6)
                if sol.success:
                    ax.plot(sol.y[0], sol.y[1], color=color, linewidth=2)
                    ax.plot(y1_0, y2_0, 'o', color=color, markersize=8)
        # Find and plot equilibrium points (simplified)
        try:
            from scipy.optimize import fsolve
            def find_equilibrium(y):
                return dydt_func(0, y)
            # Try a few initial guesses
            for y1_guess in [-1, 0, 1]:
                for y2_guess in [-1, 0, 1]:
                    try:
                        eq_point = fsolve(find_equilibrium, [y1_guess, y2_guess])
                        # Check if it's actually an equilibrium
                        residual = find_equilibrium(eq_point)
                        if np.linalg.norm(residual) < 1e-6:
                            # Check if it's in the plot range
                            if (x_range[0] <= eq_point[0] <= x_range[1] and
                                y_range[0] <= eq_point[1] <= y_range[1]):
                                ax.plot(eq_point[0], eq_point[1], 's',
                                       color=self.colors['golden_gate'],
                                       markersize=10, markeredgecolor='black')
                    except:
                        pass
        except:
            pass  # Skip equilibrium finding if it fails
        # Formatting
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, color=self.colors['berkeley_blue'])
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        self._add_berkeley_branding(fig)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    def plot_convergence_study(self,
                             step_sizes: np.ndarray,
                             errors: Dict[str, np.ndarray],
                             title: str = "Convergence Study",
                             xlabel: str = "Step Size",
                             ylabel: str = "Error",
                             log_scale: bool = True,
                             save_path: Optional[str] = None,
                             **kwargs) -> plt.Figure:
        """Plot convergence study for numerical methods.
        Args:
            step_sizes: Array of step sizes
            errors: Dictionary of method_name: error_array
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            log_scale: Use logarithmic scales
            save_path: Path to save figure
            **kwargs: Additional plot arguments
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        # Plot convergence curves
        for i, (method, error) in enumerate(errors.items()):
            color = self.method_colors.get(method,
                                         list(self.colors.values())[i % len(self.colors)])
            if log_scale:
                ax.loglog(step_sizes, error, 'o-', color=color,
                         linewidth=2, markersize=6, label=method)
            else:
                ax.plot(step_sizes, error, 'o-', color=color,
                       linewidth=2, markersize=6, label=method)
        # Add theoretical convergence lines
        if kwargs.get('show_theory', True):
            h_ref = step_sizes[0]
            err_ref = min([np.max(err) for err in errors.values()])
            # Common convergence orders
            orders = [1, 2, 4]
            order_colors = [self.colors['bay_fog'], self.colors['lap_lane'],
                           self.colors['ion']]
            for order, color in zip(orders, order_colors):
                theoretical = err_ref * (step_sizes / h_ref) ** order
                if log_scale:
                    ax.loglog(step_sizes, theoretical, '--', color=color,
                             alpha=0.7, label=f'Order {order}')
        # Formatting
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, color=self.colors['berkeley_blue'])
        ax.grid(True, alpha=0.3)
        ax.legend()
        self._add_berkeley_branding(fig)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    def plot_stability_region(self,
                            method_name: str,
                            real_range: Tuple[float, float] = (-4, 2),
                            imag_range: Tuple[float, float] = (-3, 3),
                            resolution: int = 500,
                            title: Optional[str] = None,
                            save_path: Optional[str] = None,
                            **kwargs) -> plt.Figure:
        """Plot stability region for numerical method.
        Args:
            method_name: Name of numerical method
            real_range: Range for real axis
            imag_range: Range for imaginary axis
            resolution: Grid resolution
            title: Plot title
            save_path: Path to save figure
            **kwargs: Additional plot arguments
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        # Create complex grid
        real = np.linspace(real_range[0], real_range[1], resolution)
        imag = np.linspace(imag_range[0], imag_range[1], resolution)
        R, I = np.meshgrid(real, imag)
        Z = R + 1j * I
        # Stability function for different methods
        if method_name.lower() == 'euler':
            # Forward Euler: R(z) = 1 + z
            stability = np.abs(1 + Z)
        elif method_name.lower() == 'rk4':
            # RK4: R(z) = 1 + z + z²/2 + z³/6 + z⁴/24
            stability = np.abs(1 + Z + Z**2/2 + Z**3/6 + Z**4/24)
        elif method_name.lower() == 'implicit_euler':
            # Backward Euler: R(z) = 1/(1 - z)
            stability = np.abs(1 / (1 - Z))
        elif method_name.lower() == 'crank_nicolson':
            # Crank-Nicolson: R(z) = (1 + z/2)/(1 - z/2)
            stability = np.abs((1 + Z/2) / (1 - Z/2))
        else:
            warnings.warn(f"Unknown method: {method_name}, using generic stability")
            stability = np.abs(1 + Z)
        # Plot stability region (|R(z)| ≤ 1)
        stable_region = stability <= 1
        # Contour plot
        levels = [0.5, 1.0, 1.5, 2.0]
        cs = ax.contour(R, I, stability, levels=levels,
                       colors=[self.colors['berkeley_blue']], alpha=0.6)
        ax.clabel(cs, inline=True, fontsize=10)
        # Fill stable region
        ax.contourf(R, I, stable_region, levels=[0.5, 1.5],
                   colors=[self.colors['california_gold']], alpha=0.3)
        # Highlight unit circle boundary
        ax.contour(R, I, stability, levels=[1.0],
                  colors=[self.colors['berkeley_blue']], linewidths=3)
        # Formatting
        ax.set_xlabel('Real(z)')
        ax.set_ylabel('Imag(z)')
        if title is None:
            title = f'Stability Region: {method_name.title()}'
        ax.set_title(title, color=self.colors['berkeley_blue'])
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.set_aspect('equal')
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.colors['california_gold'], alpha=0.3,
                  label='Stable Region (|R(z)| ≤ 1)'),
            plt.Line2D([0], [0], color=self.colors['berkeley_blue'],
                      linewidth=3, label='Stability Boundary')
        ]
        ax.legend(handles=legend_elements)
        self._add_berkeley_branding(fig)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    def _add_berkeley_branding(self, fig: plt.Figure):
        """Add Berkeley branding to figure."""
        fig.text(0.02, 0.02, 'UC Berkeley SciComp - ODE/PDE',
                fontsize=8, color=self.colors['berkeley_blue'], alpha=0.7)
# Convenience functions
def plot_ode_solution(t: np.ndarray, y: np.ndarray,
                     labels: Optional[List[str]] = None,
                     title: str = "ODE Solution",
                     save_path: Optional[str] = None,
                     **kwargs) -> plt.Figure:
    """Convenience function for plotting ODE solutions."""
    visualizer = ODEPDEVisualizer()
    return visualizer.plot_ode_solution(t, y, labels, title, save_path=save_path, **kwargs)
def plot_pde_solution(x: np.ndarray, u: np.ndarray,
                     y: Optional[np.ndarray] = None,
                     t: Optional[np.ndarray] = None,
                     title: str = "PDE Solution",
                     save_path: Optional[str] = None,
                     **kwargs) -> plt.Figure:
    """Convenience function for plotting PDE solutions."""
    visualizer = ODEPDEVisualizer()
    if y is None:
        # 1D PDE
        return visualizer.plot_pde_solution_1d(x, u, t, title, save_path=save_path, **kwargs)
    else:
        # 2D PDE
        return visualizer.plot_pde_solution_2d(x, y, u, title, save_path=save_path, **kwargs)
def animate_pde_evolution(x: np.ndarray, u: np.ndarray, t: np.ndarray,
                         title: str = "PDE Evolution",
                         save_path: Optional[str] = None,
                         **kwargs) -> FuncAnimation:
    """Convenience function for creating PDE animations."""
    visualizer = ODEPDEVisualizer()
    return visualizer.animate_pde_evolution(x, u, t, title, save_path, **kwargs)
def create_phase_portrait(dydt_func: Callable,
                         trajectories: Optional[List[Tuple[float, float]]] = None,
                         title: str = "Phase Portrait",
                         save_path: Optional[str] = None,
                         **kwargs) -> plt.Figure:
    """Convenience function for creating phase portraits."""
    visualizer = ODEPDEVisualizer()
    return visualizer.create_phase_portrait(dydt_func, trajectories=trajectories,
                                          title=title, save_path=save_path, **kwargs)