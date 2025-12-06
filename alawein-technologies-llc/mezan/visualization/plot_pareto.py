"""
Pareto Front Visualization for Librex.Evo (Multi-Objective Optimization)

Publication-quality Pareto front plots for 2D, 3D, and many-objective problems.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path


class ParetoPlotter:
    """
    Visualize Pareto fronts for multi-objective optimization

    Supports:
    - 2D Pareto fronts (2 objectives)
    - 3D Pareto fronts (3 objectives)
    - Parallel coordinates (many objectives)
    - Hypervolume visualization
    """

    def __init__(self, style: str = "seaborn-v0_8"):
        plt.style.use(style)
        self.colors = plt.cm.viridis

    def plot_2d_pareto(
        self,
        pareto_front: np.ndarray,
        all_solutions: Optional[np.ndarray] = None,
        obj1_name: str = "Objective 1",
        obj2_name: str = "Objective 2",
        title: str = "Pareto Front (2D)",
        reference_point: Optional[Tuple[float, float]] = None,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot 2D Pareto front

        Args:
            pareto_front: Nx2 array of Pareto-optimal objective values
            all_solutions: Mx2 array of all evaluated solutions (for context)
            obj1_name: Name of first objective
            obj2_name: Name of second objective
            title: Plot title
            reference_point: Reference point for hypervolume
            save_path: Save path

        Returns:
            Figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot all solutions (if provided)
        if all_solutions is not None:
            ax.scatter(all_solutions[:, 0], all_solutions[:, 1],
                      c='lightgray', alpha=0.5, s=30, label='All Solutions')

        # Plot Pareto front
        # Sort for line plot
        sorted_idx = np.argsort(pareto_front[:, 0])
        pareto_sorted = pareto_front[sorted_idx]

        ax.plot(pareto_sorted[:, 0], pareto_sorted[:, 1],
               'b-', linewidth=2, alpha=0.7, label='Pareto Front')
        ax.scatter(pareto_front[:, 0], pareto_front[:, 1],
                  c='red', s=100, marker='*', edgecolors='black',
                  linewidths=1.5, label='Pareto-Optimal Solutions', zorder=5)

        # Reference point (for hypervolume)
        if reference_point:
            ax.scatter([reference_point[0]], [reference_point[1]],
                      c='green', s=200, marker='X',
                      label=f'Reference Point ({reference_point[0]:.1f}, {reference_point[1]:.1f})',
                      zorder=6)

            # Shade hypervolume
            x_fill = np.append(pareto_sorted[:, 0], reference_point[0])
            y_fill = np.append(pareto_sorted[:, 1], reference_point[1])
            ax.fill(x_fill, y_fill, alpha=0.2, color='blue', label='Hypervolume')

        ax.set_xlabel(obj1_name, fontsize=14, fontweight='bold')
        ax.set_ylabel(obj2_name, fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(fontsize=11, loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_3d_pareto(
        self,
        pareto_front: np.ndarray,
        all_solutions: Optional[np.ndarray] = None,
        obj_names: Optional[List[str]] = None,
        title: str = "Pareto Front (3D)",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot 3D Pareto front

        Args:
            pareto_front: Nx3 array of Pareto-optimal solutions
            all_solutions: Mx3 array of all solutions
            obj_names: Names of three objectives
            title: Plot title
            save_path: Save path

        Returns:
            Figure
        """
        if obj_names is None:
            obj_names = ["Objective 1", "Objective 2", "Objective 3"]

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot all solutions
        if all_solutions is not None:
            ax.scatter(all_solutions[:, 0], all_solutions[:, 1], all_solutions[:, 2],
                      c='lightgray', alpha=0.3, s=20, label='All Solutions')

        # Plot Pareto front
        sc = ax.scatter(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2],
                       c=pareto_front[:, 2], cmap=self.colors,
                       s=150, marker='*', edgecolors='black', linewidths=1.5,
                       label='Pareto-Optimal', zorder=5)

        # Colorbar
        cbar = plt.colorbar(sc, ax=ax, shrink=0.7, aspect=10)
        cbar.set_label(obj_names[2], fontsize=12, fontweight='bold')

        ax.set_xlabel(obj_names[0], fontsize=12, fontweight='bold')
        ax.set_ylabel(obj_names[1], fontsize=12, fontweight='bold')
        ax.set_zlabel(obj_names[2], fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=11, loc='best')

        # Rotate for better view
        ax.view_init(elev=20, azim=45)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_parallel_coordinates(
        self,
        pareto_front: np.ndarray,
        obj_names: Optional[List[str]] = None,
        title: str = "Pareto Front (Parallel Coordinates)",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Parallel coordinates plot for many-objective problems (>3 objectives)

        Args:
            pareto_front: NxM array (M objectives)
            obj_names: Names of M objectives
            title: Plot title
            save_path: Save path

        Returns:
            Figure
        """
        n_solutions, n_objectives = pareto_front.shape

        if obj_names is None:
            obj_names = [f"Obj {i+1}" for i in range(n_objectives)]

        fig, ax = plt.subplots(figsize=(14, 7))

        # Normalize each objective to [0, 1] for better visualization
        norm_front = (pareto_front - pareto_front.min(axis=0)) / \
                     (pareto_front.max(axis=0) - pareto_front.min(axis=0) + 1e-10)

        x_pos = np.arange(n_objectives)

        # Plot each solution as a line
        for i in range(n_solutions):
            color = self.colors(i / n_solutions)
            ax.plot(x_pos, norm_front[i, :], color=color, alpha=0.6, linewidth=1.5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(obj_names, rotation=45, ha='right', fontsize=11)
        ax.set_ylabel('Normalized Objective Value', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_hypervolume_evolution(
        self,
        generations: List[int],
        hypervolumes: List[float],
        title: str = "Hypervolume Evolution",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot hypervolume indicator over generations

        Args:
            generations: Generation numbers
            hypervolumes: Hypervolume at each generation
            title: Plot title
            save_path: Save path

        Returns:
            Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(generations, hypervolumes, 'b-', linewidth=2.5, marker='o', markersize=6)

        ax.set_xlabel('Generation', fontsize=14, fontweight='bold')
        ax.set_ylabel('Hypervolume Indicator', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Annotate final value
        final_hv = hypervolumes[-1]
        ax.annotate(f'Final HV: {final_hv:.4f}',
                   xy=(generations[-1], final_hv),
                   xytext=(generations[-1]*0.7, final_hv*0.95),
                   fontsize=12, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', lw=2),
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_attainment_surface(
        self,
        pareto_fronts_across_runs: List[np.ndarray],
        quantile: float = 0.5,
        obj1_name: str = "Objective 1",
        obj2_name: str = "Objective 2",
        title: str = "Empirical Attainment Function",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot empirical attainment surface (for 2D problems)

        Shows the median attainment surface across multiple runs.

        Args:
            pareto_fronts_across_runs: List of Pareto fronts from multiple runs
            quantile: Quantile to plot (0.5 = median)
            obj1_name: First objective name
            obj2_name: Second objective name
            title: Plot title
            save_path: Save path

        Returns:
            Figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot individual runs in light colors
        for i, front in enumerate(pareto_fronts_across_runs):
            sorted_idx = np.argsort(front[:, 0])
            front_sorted = front[sorted_idx]
            ax.plot(front_sorted[:, 0], front_sorted[:, 1],
                   'gray', alpha=0.3, linewidth=1)

        # Compute quantile attainment surface (simplified)
        all_points = np.vstack(pareto_fronts_across_runs)
        # Simple approach: grid and compute quantile at each grid point
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        x_grid = np.linspace(x_min, x_max, 100)

        y_quantile = []
        for x in x_grid:
            # Find y values at or near this x
            nearby = all_points[np.abs(all_points[:, 0] - x) < (x_max - x_min) / 50]
            if len(nearby) > 0:
                y_quantile.append(np.quantile(nearby[:, 1], quantile))
            else:
                y_quantile.append(np.nan)

        y_quantile = np.array(y_quantile)
        valid = ~np.isnan(y_quantile)

        ax.plot(x_grid[valid], y_quantile[valid],
               'r-', linewidth=3, label=f'{int(quantile*100)}% Attainment Surface')

        ax.set_xlabel(obj1_name, fontsize=14, fontweight='bold')
        ax.set_ylabel(obj2_name, fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


# Convenience functions

def quick_pareto_2d(
    pareto_front: np.ndarray,
    obj1: str = "Speed",
    obj2: str = "Quality",
    output: str = "pareto_2d.png"
):
    """Quick 2D Pareto plot"""
    plotter = ParetoPlotter()
    plotter.plot_2d_pareto(pareto_front, obj1_name=obj1, obj2_name=obj2, save_path=Path(output))
    print(f"Pareto front saved to {output}")


def quick_pareto_3d(
    pareto_front: np.ndarray,
    objectives: List[str] = None,
    output: str = "pareto_3d.png"
):
    """Quick 3D Pareto plot"""
    plotter = ParetoPlotter()
    plotter.plot_3d_pareto(pareto_front, obj_names=objectives, save_path=Path(output))
    print(f"3D Pareto front saved to {output}")
