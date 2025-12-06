"""
Visualization tools for quantum states and materials properties.

Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
License: MIT License © 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
import warnings

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def setup_plotting_style():
    """Configure matplotlib/seaborn for publication-ready plots."""
    if HAS_SEABORN:
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


def plot_quantum_state(
    state: np.ndarray,
    title: str = "Quantum State",
    basis_labels: Optional[List[str]] = None
) -> plt.Figure:
    """
    Visualize a quantum state vector.
    
    Args:
        state: Quantum state vector
        title: Plot title
        basis_labels: Labels for basis states
    
    Returns:
        Matplotlib figure
    """
    state = state.flatten()
    n_basis = len(state)
    
    if basis_labels is None:
        n_qubits = int(np.log2(n_basis))
        basis_labels = [format(i, f'0{n_qubits}b') for i in range(n_basis)]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Probability distribution
    probabilities = np.abs(state) ** 2
    ax1.bar(range(n_basis), probabilities, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Basis State')
    ax1.set_ylabel('Probability')
    ax1.set_title(f'{title} - Probability Distribution')
    ax1.set_xticks(range(n_basis))
    ax1.set_xticklabels(basis_labels, rotation=45 if n_basis > 8 else 0)
    ax1.grid(True, alpha=0.3)
    
    # Phase information
    phases = np.angle(state)
    colors = plt.cm.hsv((phases + np.pi) / (2 * np.pi))
    ax2.bar(range(n_basis), probabilities, color=colors)
    ax2.set_xlabel('Basis State')
    ax2.set_ylabel('Amplitude')
    ax2.set_title(f'{title} - Phase Information')
    ax2.set_xticks(range(n_basis))
    ax2.set_xticklabels(basis_labels, rotation=45 if n_basis > 8 else 0)
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar for phase
    sm = plt.cm.ScalarMappable(cmap=plt.cm.hsv, norm=plt.Normalize(vmin=-np.pi, vmax=np.pi))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, orientation='horizontal', pad=0.1, fraction=0.05)
    cbar.set_label('Phase (radians)')
    
    plt.tight_layout()
    return fig


def plot_measurement_results(
    counts: Dict[str, int],
    title: str = "Measurement Results"
) -> plt.Figure:
    """
    Plot measurement count results.
    
    Args:
        counts: Dictionary of measurement counts
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = list(counts.keys())
    values = list(counts.values())
    total = sum(values)
    
    # Sort by count
    sorted_pairs = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
    labels, values = zip(*sorted_pairs)
    
    # Plot bars
    bars = ax.bar(range(len(labels)), values, color='teal', alpha=0.7)
    
    # Add percentage labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        percentage = (val / total) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                f'{percentage:.1f}%', ha='center', va='bottom')
    
    ax.set_xlabel('Measurement Outcome')
    ax.set_ylabel('Counts')
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45 if len(labels) > 8 else 0)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_convergence(
    iterations: List[float],
    values: List[float],
    title: str = "Convergence Plot",
    ylabel: str = "Value",
    target_value: Optional[float] = None
) -> plt.Figure:
    """
    Plot convergence of an optimization process.
    
    Args:
        iterations: List of iteration numbers
        values: List of values at each iteration
        title: Plot title
        ylabel: Y-axis label
        target_value: Optional target value to show as horizontal line
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(iterations, values, 'o-', linewidth=2, markersize=6, color='darkblue')
    
    if target_value is not None:
        ax.axhline(y=target_value, color='red', linestyle='--', linewidth=2,
                   label=f'Target: {target_value:.4f}')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if target_value is not None:
        ax.legend()
    
    plt.tight_layout()
    return fig


def plot_band_structure(
    k_points: np.ndarray,
    energies: np.ndarray,
    title: str = "Band Structure",
    fermi_energy: Optional[float] = None
) -> plt.Figure:
    """
    Plot electronic band structure.
    
    Args:
        k_points: Array of k-points
        energies: Energy eigenvalues (shape: n_k x n_bands)
        title: Plot title
        fermi_energy: Optional Fermi energy level
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot each band
    for band_idx in range(energies.shape[1]):
        ax.plot(k_points, energies[:, band_idx], 'b-', linewidth=2)
    
    if fermi_energy is not None:
        ax.axhline(y=fermi_energy, color='red', linestyle='--', linewidth=1.5,
                   label=f'E_F = {fermi_energy:.2f} eV')
    
    ax.set_xlabel('k-point')
    ax.set_ylabel('Energy (eV)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if fermi_energy is not None:
        ax.legend()
    
    plt.tight_layout()
    return fig


def plot_crystal_structure(
    positions: np.ndarray,
    atomic_numbers: List[int],
    lattice_vectors: Optional[np.ndarray] = None,
    title: str = "Crystal Structure"
) -> plt.Figure:
    """
    Plot 3D crystal structure.
    
    Args:
        positions: Atomic positions (N x 3)
        atomic_numbers: List of atomic numbers
        lattice_vectors: Optional lattice vectors (3 x 3)
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color map for different elements
    unique_z = np.unique(atomic_numbers)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_z)))
    color_map = {z: colors[i] for i, z in enumerate(unique_z)}
    
    # Element symbols (simplified)
    element_symbols = {
        1: 'H', 6: 'C', 7: 'N', 8: 'O', 14: 'Si', 16: 'S',
        42: 'Mo', 74: 'W', 34: 'Se'
    }
    
    # Plot atoms
    for i, (pos, z) in enumerate(zip(positions, atomic_numbers)):
        color = color_map[z]
        size = 100 + z * 5  # Size proportional to atomic number
        ax.scatter(pos[0], pos[1], pos[2], c=[color], s=size, alpha=0.8,
                  edgecolors='black', linewidths=1)
        
        # Add element label
        if z in element_symbols:
            ax.text(pos[0], pos[1], pos[2], element_symbols[z],
                   fontsize=10, ha='center', va='center')
    
    # Plot unit cell if lattice vectors provided
    if lattice_vectors is not None:
        # Define corners of unit cell
        corners = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ])
        
        # Transform to Cartesian coordinates
        corners_cart = np.dot(corners, lattice_vectors)
        
        # Define edges of unit cell
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
        ]
        
        # Plot edges
        for edge in edges:
            points = corners_cart[edge]
            ax.plot3D(*points.T, 'k-', linewidth=1, alpha=0.3)
    
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(title)
    
    plt.tight_layout()
    return fig


def plot_ml_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "ML Model Performance",
    xlabel: str = "True Value",
    ylabel: str = "Predicted Value"
) -> plt.Figure:
    """
    Plot machine learning model performance.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot
    ax1.scatter(y_true, y_pred, alpha=0.5, s=30, color='steelblue', edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Add metrics text
    metrics_text = f'R² = {r2:.3f}\nMAE = {mae:.3f}\nRMSE = {rmse:.3f}'
    ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            verticalalignment='top', fontsize=11)
    
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(f'{title} - Predictions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Residual plot
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.5, s=30, color='coral', edgecolors='black', linewidth=0.5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Add ±1 std lines
    std_residual = np.std(residuals)
    ax2.axhline(y=std_residual, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(y=-std_residual, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    ax2.set_xlabel('Predicted Value')
    ax2.set_ylabel('Residual')
    ax2.set_title(f'{title} - Residuals')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix"
) -> plt.Figure:
    """
    Plot confusion matrix for classification results.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional names for classes
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot confusion matrix
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Labels
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=class_names,
           yticklabels=class_names,
           xlabel='Predicted Label',
           ylabel='True Label',
           title=title)
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    return fig


# TODO: Add more specialized plotting functions for materials science
# Ideas to implement:
# - Density of states plots
# - Phonon dispersion curves  
# - Property correlation heatmaps
# - Interactive 3D structure plots (maybe with plotly?)
#
# RESEARCH NOTES:
# - Check ASE visualization functions for inspiration
# - Look into plotly for interactive plots in Jupyter notebooks
# - Consider adding Brillouin zone plotting functionality
#
# INCOMPLETE: Started working on DOS plotting but need to research format
def plot_density_of_states():
    """Plot electronic density of states - TODO: implement"""
    # Need to research standard DOS data format
    # Usually energy vs DOS(E), but need to handle spin-polarized case
    pass