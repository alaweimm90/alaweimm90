"""
Convergence Visualization for MEZAN Libria Solvers

Creates publication-quality convergence plots for optimization algorithms.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class ConvergencePlotter:
    """
    Visualize algorithm convergence for all Libria solvers

    Features:
    - Real-time convergence plots
    - Multi-algorithm comparison
    - Animated convergence
    - Publication-quality figures
    """

    def __init__(self, style: str = "seaborn-v0_8"):
        """
        Initialize plotter

        Args:
            style: Matplotlib style ('seaborn-v0_8', 'ggplot', 'bmh', etc.)
        """
        plt.style.use(style)
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))

    def plot_single_convergence(
        self,
        iterations: List[int],
        objective_values: List[float],
        algorithm_name: str,
        best_known: Optional[float] = None,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot convergence for a single algorithm run

        Args:
            iterations: List of iteration numbers
            objective_values: Objective values at each iteration
            algorithm_name: Name of algorithm (e.g., "Simulated Annealing")
            best_known: Best known/optimal value (for gap visualization)
            save_path: Path to save figure (PNG/PDF)

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot objective value
        ax.plot(iterations, objective_values, 'b-', linewidth=2, label='Objective Value')

        # Plot best known value if provided
        if best_known is not None:
            ax.axhline(y=best_known, color='r', linestyle='--',
                      linewidth=2, label=f'Best Known ({best_known:.2f})')

            # Shade gap
            ax.fill_between(iterations, objective_values, best_known,
                           alpha=0.2, color='orange', label='Optimality Gap')

        # Styling
        ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
        ax.set_ylabel('Objective Value', fontsize=14, fontweight='bold')
        ax.set_title(f'{algorithm_name} - Convergence', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_multi_algorithm_comparison(
        self,
        results: Dict[str, Tuple[List[int], List[float]]],
        title: str = "Algorithm Comparison",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Compare convergence of multiple algorithms

        Args:
            results: Dict mapping algorithm name to (iterations, objectives)
            title: Plot title
            save_path: Save path

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 7))

        for idx, (alg_name, (iters, objs)) in enumerate(results.items()):
            color = self.colors[idx % len(self.colors)]
            ax.plot(iters, objs, color=color, linewidth=2.5,
                   label=alg_name, marker='o', markersize=4, markevery=max(1, len(iters)//10))

        ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
        ax.set_ylabel('Objective Value', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(fontsize=11, loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)

        # Log scale if values span multiple orders of magnitude
        if ax.get_ylim()[1] / ax.get_ylim()[0] > 100:
            ax.set_yscale('log')
            ax.set_ylabel('Objective Value (log scale)', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_convergence_with_variance(
        self,
        iterations: List[int],
        mean_values: List[float],
        std_values: List[float],
        algorithm_name: str,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot convergence with confidence intervals (from multiple runs)

        Args:
            iterations: Iteration numbers
            mean_values: Mean objective values
            std_values: Standard deviation at each iteration
            algorithm_name: Algorithm name
            save_path: Save path

        Returns:
            Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        mean_values = np.array(mean_values)
        std_values = np.array(std_values)
        iterations = np.array(iterations)

        # Plot mean
        ax.plot(iterations, mean_values, 'b-', linewidth=2.5, label='Mean')

        # Confidence intervals (mean ± std)
        ax.fill_between(iterations,
                       mean_values - std_values,
                       mean_values + std_values,
                       alpha=0.3, color='blue', label='±1 std')

        ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
        ax.set_ylabel('Objective Value', fontsize=14, fontweight='bold')
        ax.set_title(f'{algorithm_name} - Convergence with Variance',
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def create_animated_convergence(
        self,
        iterations: List[int],
        objective_values: List[float],
        algorithm_name: str,
        save_path: Optional[Path] = None,
        fps: int = 10,
    ) -> animation.FuncAnimation:
        """
        Create animated convergence plot

        Args:
            iterations: Iterations
            objective_values: Objectives
            algorithm_name: Algorithm name
            save_path: Save path (GIF or MP4)
            fps: Frames per second

        Returns:
            Animation object
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        line, = ax.plot([], [], 'b-', linewidth=2)
        point, = ax.plot([], [], 'ro', markersize=8)

        ax.set_xlim(min(iterations), max(iterations))
        ax.set_ylim(min(objective_values) * 0.95, max(objective_values) * 1.05)
        ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
        ax.set_ylabel('Objective Value', fontsize=14, fontweight='bold')
        ax.set_title(f'{algorithm_name} - Convergence Animation',
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)

        def init():
            line.set_data([], [])
            point.set_data([], [])
            return line, point

        def animate(frame):
            x_data = iterations[:frame+1]
            y_data = objective_values[:frame+1]
            line.set_data(x_data, y_data)
            if frame < len(iterations):
                point.set_data([iterations[frame]], [objective_values[frame]])
            return line, point

        anim = animation.FuncAnimation(
            fig, animate, init_func=init,
            frames=len(iterations), interval=1000//fps, blit=True
        )

        if save_path:
            writer = animation.PillowWriter(fps=fps) if str(save_path).endswith('.gif') else 'ffmpeg'
            anim.save(save_path, writer=writer)

        return anim

    def plot_temperature_schedule(
        self,
        iterations: List[int],
        temperatures: List[float],
        objective_values: List[float],
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot simulated annealing temperature schedule with convergence

        Args:
            iterations: Iterations
            temperatures: Temperature at each iteration
            objective_values: Objectives
            save_path: Save path

        Returns:
            Figure with dual y-axes
        """
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Objective on left axis
        color = 'tab:blue'
        ax1.set_xlabel('Iteration', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Objective Value', fontsize=14, fontweight='bold', color=color)
        ax1.plot(iterations, objective_values, color=color, linewidth=2, label='Objective')
        ax1.tick_params(axis='y', labelcolor=color)

        # Temperature on right axis
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Temperature', fontsize=14, fontweight='bold', color=color)
        ax2.plot(iterations, temperatures, color=color, linewidth=2,
                linestyle='--', label='Temperature')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.suptitle('Simulated Annealing - Temperature Schedule',
                    fontsize=16, fontweight='bold')

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=12)

        ax1.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


# Convenience functions

def quick_convergence_plot(
    iterations: List[int],
    objectives: List[float],
    title: str = "Convergence",
    output_file: str = "convergence.png"
):
    """Quick convergence plot with sensible defaults"""
    plotter = ConvergencePlotter()
    plotter.plot_single_convergence(
        iterations, objectives, title, save_path=Path(output_file)
    )
    print(f"Convergence plot saved to {output_file}")


def compare_algorithms(
    results: Dict[str, Tuple[List[int], List[float]]],
    output_file: str = "algorithm_comparison.png"
):
    """Quick multi-algorithm comparison"""
    plotter = ConvergencePlotter()
    plotter.plot_multi_algorithm_comparison(
        results, save_path=Path(output_file)
    )
    print(f"Comparison plot saved to {output_file}")
