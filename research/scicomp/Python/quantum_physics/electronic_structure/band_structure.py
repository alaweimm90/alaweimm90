#!/usr/bin/env python3
"""
Electronic Band Structure Calculations
Implementation of tight-binding models and band structure calculations
for crystalline materials, including high-symmetry path generation
and Berkeley-styled visualization.
Key Features:
- Tight-binding Hamiltonian construction
- k-space sampling and high-symmetry paths
- Band structure calculation with exact diagonalization
- Effective mass calculations
- Strain effects on band structure
- Publication-quality plotting with Berkeley theme
Mathematical Foundation:
The tight-binding Hamiltonian in k-space is:
H(k) = Σᵢⱼ tᵢⱼ exp(ik⋅Rᵢⱼ) |i⟩⟨j|
where tᵢⱼ are hopping parameters and Rᵢⱼ are lattice vectors.
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
import numpy as np
from scipy import linalg
from typing import Dict, List, Tuple, Optional, Callable, Union
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from ...utils.constants import hbar, me, eV_to_J
from ...visualization.berkeley_style import BerkeleyPlot, BERKELEY_BLUE, CALIFORNIA_GOLD
class BandStructureCalculator:
    """
    Electronic band structure calculator using tight-binding models.
    Supports various crystal structures and provides tools for band structure
    visualization, effective mass calculation, and strain engineering.
    """
    def __init__(self,
                 lattice_vectors: np.ndarray,
                 basis_positions: np.ndarray,
                 hopping_parameters: Dict[str, float],
                 onsite_energies: Optional[Dict[int, float]] = None):
        """
        Initialize band structure calculator.
        Parameters
        ----------
        lattice_vectors : ndarray, shape (2, 2) or (3, 3)
            Lattice vectors defining the unit cell
        basis_positions : ndarray, shape (n_atoms, 2) or (n_atoms, 3)
            Positions of atoms in the unit cell
        hopping_parameters : dict
            Hopping parameters between orbitals
        onsite_energies : dict, optional
            On-site energies for each orbital
        """
        self.lattice_vectors = np.array(lattice_vectors)
        self.basis_positions = np.array(basis_positions)
        self.hopping_parameters = hopping_parameters
        self.onsite_energies = onsite_energies or {}
        self.dimension = self.lattice_vectors.shape[0]
        self.num_atoms = len(self.basis_positions)
        # Calculate reciprocal lattice vectors
        if self.dimension == 2:
            self.reciprocal_vectors = self._calculate_reciprocal_2d()
        else:
            self.reciprocal_vectors = self._calculate_reciprocal_3d()
        # Storage for calculated bands
        self.k_points = None
        self.energies = None
        self.eigenvectors = None
    def _calculate_reciprocal_2d(self) -> np.ndarray:
        """Calculate reciprocal lattice vectors for 2D system."""
        a1, a2 = self.lattice_vectors
        # 2π factor for reciprocal lattice
        factor = 2 * np.pi / np.cross(a1, a2)
        b1 = factor * np.array([a2[1], -a2[0]])
        b2 = factor * np.array([-a1[1], a1[0]])
        return np.array([b1, b2])
    def _calculate_reciprocal_3d(self) -> np.ndarray:
        """Calculate reciprocal lattice vectors for 3D system."""
        a1, a2, a3 = self.lattice_vectors
        volume = np.dot(a1, np.cross(a2, a3))
        factor = 2 * np.pi / volume
        b1 = factor * np.cross(a2, a3)
        b2 = factor * np.cross(a3, a1)
        b3 = factor * np.cross(a1, a2)
        return np.array([b1, b2, b3])
    def construct_hamiltonian(self, k_point: np.ndarray) -> np.ndarray:
        """
        Construct tight-binding Hamiltonian for given k-point.
        Parameters
        ----------
        k_point : ndarray
            k-point in reciprocal space
        Returns
        -------
        ndarray
            Hamiltonian matrix H(k)
        """
        H = np.zeros((self.num_atoms, self.num_atoms), dtype=complex)
        # On-site energies (diagonal terms)
        for i in range(self.num_atoms):
            H[i, i] = self.onsite_energies.get(i, 0.0)
        # Hopping terms
        for (i, j), t_ij in self.hopping_parameters.items():
            if i != j:  # Off-diagonal terms
                # Calculate lattice vector between atoms
                r_ij = self.basis_positions[j] - self.basis_positions[i]
                # Add contributions from periodic images
                phase = np.exp(1j * np.dot(k_point, r_ij))
                H[i, j] += t_ij * phase
                H[j, i] += np.conj(t_ij * phase)  # Hermitian conjugate
        return H
    def calculate_bands(self,
                       k_points: np.ndarray,
                       num_bands: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate electronic band structure.
        Parameters
        ----------
        k_points : ndarray, shape (n_kpoints, dim)
            k-points for band calculation
        num_bands : int, optional
            Number of bands to calculate. If None, calculates all bands
        Returns
        -------
        energies : ndarray, shape (n_kpoints, n_bands)
            Band energies
        eigenvectors : ndarray, shape (n_kpoints, n_bands, n_atoms)
            Corresponding eigenvectors
        """
        n_kpoints = len(k_points)
        if num_bands is None:
            num_bands = self.num_atoms
        energies = np.zeros((n_kpoints, num_bands))
        eigenvectors = np.zeros((n_kpoints, num_bands, self.num_atoms), dtype=complex)
        for i, k in enumerate(k_points):
            H_k = self.construct_hamiltonian(k)
            # Solve eigenvalue problem
            eigenvals, eigenvecs = linalg.eigh(H_k)
            # Sort by energy (already sorted by eigh)
            energies[i, :] = eigenvals[:num_bands]
            eigenvectors[i, :, :] = eigenvecs[:, :num_bands].T
        self.k_points = k_points
        self.energies = energies
        self.eigenvectors = eigenvectors
        return energies, eigenvectors
    def generate_high_symmetry_path(self,
                                  crystal_system: str,
                                  n_points: int = 100) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Generate high-symmetry k-point path for band structure plotting.
        Parameters
        ----------
        crystal_system : str
            Crystal system ('square', 'hexagonal', 'cubic')
        n_points : int, default 100
            Number of k-points along the path
        Returns
        -------
        k_path : ndarray
            k-points along high-symmetry path
        labels : list
            High-symmetry point labels
        k_distances : ndarray
            Distances along k-path for plotting
        """
        if crystal_system.lower() == 'square':
            return self._square_lattice_path(n_points)
        elif crystal_system.lower() == 'hexagonal':
            return self._hexagonal_lattice_path(n_points)
        elif crystal_system.lower() == 'cubic':
            return self._cubic_lattice_path(n_points)
        else:
            raise ValueError(f"Unknown crystal system: {crystal_system}")
    def _square_lattice_path(self, n_points: int) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Generate path for square lattice: Γ → X → M → Γ."""
        # High-symmetry points in reciprocal space
        gamma = np.array([0.0, 0.0])
        X = np.array([np.pi, 0.0])
        M = np.array([np.pi, np.pi])
        # Create path segments
        path_segments = [
            (gamma, X, 'Γ', 'X'),
            (X, M, 'X', 'M'),
            (M, gamma, 'M', 'Γ')
        ]
        return self._create_path_from_segments(path_segments, n_points)
    def _hexagonal_lattice_path(self, n_points: int) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Generate path for hexagonal lattice: Γ → M → K → Γ."""
        # High-symmetry points
        gamma = np.array([0.0, 0.0])
        M = np.array([0.0, 2*np.pi/3])
        K = np.array([2*np.pi/3, 2*np.pi/3])
        path_segments = [
            (gamma, M, 'Γ', 'M'),
            (M, K, 'M', 'K'),
            (K, gamma, 'K', 'Γ')
        ]
        return self._create_path_from_segments(path_segments, n_points)
    def _cubic_lattice_path(self, n_points: int) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Generate path for cubic lattice: Γ → X → M → R → Γ."""
        gamma = np.array([0.0, 0.0, 0.0])
        X = np.array([np.pi, 0.0, 0.0])
        M = np.array([np.pi, np.pi, 0.0])
        R = np.array([np.pi, np.pi, np.pi])
        path_segments = [
            (gamma, X, 'Γ', 'X'),
            (X, M, 'X', 'M'),
            (M, R, 'M', 'R'),
            (R, gamma, 'R', 'Γ')
        ]
        return self._create_path_from_segments(path_segments, n_points)
    def _create_path_from_segments(self,
                                 path_segments: List[Tuple],
                                 n_points: int) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Create k-path from segments."""
        total_segments = len(path_segments)
        points_per_segment = n_points // total_segments
        k_path = []
        labels = []
        k_distances = []
        current_distance = 0.0
        for i, (start, end, start_label, end_label) in enumerate(path_segments):
            if i == 0:
                labels.append(start_label)
            # Create points along segment
            segment_points = np.linspace(start, end, points_per_segment, endpoint=False)
            if i == len(path_segments) - 1:  # Include endpoint for last segment
                segment_points = np.linspace(start, end, points_per_segment + 1)
            # Calculate distances
            for j, k_point in enumerate(segment_points):
                if len(k_path) > 0:
                    dk = np.linalg.norm(k_point - k_path[-1])
                    current_distance += dk
                k_path.append(k_point)
                k_distances.append(current_distance)
            # Add segment end label
            if i < len(path_segments) - 1:
                labels.append(end_label)
            else:
                labels.append(end_label)
        return np.array(k_path), labels, np.array(k_distances)
    def calculate_effective_mass(self,
                                band_index: int,
                                k_point: np.ndarray,
                                direction: np.ndarray,
                                dk: float = 1e-4) -> float:
        """
        Calculate effective mass using finite differences.
        Parameters
        ----------
        band_index : int
            Band index
        k_point : ndarray
            k-point around which to calculate effective mass
        direction : ndarray
            Direction in k-space
        dk : float, default 1e-4
            k-space step size
        Returns
        -------
        float
            Effective mass in units of electron mass
        """
        direction = direction / np.linalg.norm(direction)
        # Calculate energies at k ± dk
        k_plus = k_point + dk * direction
        k_minus = k_point - dk * direction
        E_plus = linalg.eigvals(self.construct_hamiltonian(k_plus))[band_index]
        E_minus = linalg.eigvals(self.construct_hamiltonian(k_minus))[band_index]
        # Second derivative: d²E/dk²
        d2E_dk2 = (E_plus - 2*linalg.eigvals(self.construct_hamiltonian(k_point))[band_index] + E_minus) / dk**2
        # Effective mass: m* = ħ² / (d²E/dk²)
        m_eff = hbar**2 / (d2E_dk2 * eV_to_J)
        return m_eff / me  # In units of electron mass
    def plot_band_structure(self,
                           k_path: Optional[np.ndarray] = None,
                           labels: Optional[List[str]] = None,
                           k_distances: Optional[np.ndarray] = None,
                           energy_range: Optional[Tuple[float, float]] = None,
                           fermi_energy: Optional[float] = None,
                           output_dir: Optional[Path] = None) -> plt.Figure:
        """
        Plot band structure with Berkeley styling.
        Parameters
        ----------
        k_path : ndarray, optional
            k-points (uses stored if None)
        labels : list, optional
            High-symmetry point labels
        k_distances : ndarray, optional
            Distances for x-axis
        energy_range : tuple, optional
            Energy range for y-axis
        fermi_energy : float, optional
            Fermi energy to plot as horizontal line
        output_dir : Path, optional
            Directory to save figure
        Returns
        -------
        Figure
            Matplotlib figure object
        """
        if self.energies is None:
            raise ValueError("No band structure calculated. Run calculate_bands() first.")
        if k_distances is None:
            k_distances = np.arange(len(self.energies))
        # Create Berkeley-styled plot
        berkeley_plot = BerkeleyPlot(figsize=(10, 8))
        fig, ax = berkeley_plot.create_figure()
        # Plot all bands
        n_bands = self.energies.shape[1]
        for band in range(n_bands):
            ax.plot(k_distances, self.energies[:, band],
                   color=BERKELEY_BLUE, linewidth=2, alpha=0.8)
        # Plot Fermi energy if provided
        if fermi_energy is not None:
            ax.axhline(y=fermi_energy, color=CALIFORNIA_GOLD,
                      linestyle='--', linewidth=2,
                      label=f'E_F = {fermi_energy:.2f} eV')
            ax.legend()
        # Set labels and formatting
        ax.set_xlabel('k-path')
        ax.set_ylabel('Energy (eV)')
        ax.set_title('Electronic Band Structure', fontweight='bold', color=BERKELEY_BLUE)
        # Add high-symmetry point labels
        if labels is not None:
            # Find positions for labels
            label_positions = []
            if len(labels) > 1:
                segment_length = len(k_distances) // (len(labels) - 1)
                for i in range(len(labels)):
                    if i == len(labels) - 1:
                        label_positions.append(k_distances[-1])
                    else:
                        label_positions.append(k_distances[i * segment_length])
            ax.set_xticks(label_positions)
            ax.set_xticklabels(labels)
            # Add vertical lines at high-symmetry points
            for pos in label_positions[1:-1]:  # Skip first and last
                ax.axvline(x=pos, color='gray', linestyle='-', alpha=0.5, linewidth=1)
        # Set energy range if provided
        if energy_range is not None:
            ax.set_ylim(energy_range)
        ax.grid(True, alpha=0.3)
        # Save figure if requested
        if output_dir:
            berkeley_plot.save_figure(output_dir / "band_structure.png")
        return fig
    def apply_strain(self, strain_tensor: np.ndarray) -> 'BandStructureCalculator':
        """
        Apply strain to the crystal and return new band structure calculator.
        Parameters
        ----------
        strain_tensor : ndarray
            Strain tensor (2x2 or 3x3)
        Returns
        -------
        BandStructureCalculator
            New calculator with strained lattice
        """
        # Apply strain to lattice vectors
        strained_lattice = (np.eye(self.dimension) + strain_tensor) @ self.lattice_vectors
        # Modify hopping parameters based on strain (simple model)
        strained_hoppings = {}
        for (i, j), t_ij in self.hopping_parameters.items():
            if i != j:
                # Calculate bond vector
                r_ij = self.basis_positions[j] - self.basis_positions[i]
                r_ij_strained = (np.eye(self.dimension) + strain_tensor) @ r_ij
                # Scale hopping with inverse bond length (simple model)
                bond_change = np.linalg.norm(r_ij_strained) / np.linalg.norm(r_ij)
                strained_hoppings[(i, j)] = t_ij / bond_change
            else:
                strained_hoppings[(i, j)] = t_ij
        return BandStructureCalculator(
            strained_lattice,
            self.basis_positions,
            strained_hoppings,
            self.onsite_energies
        )
# Convenience functions for common materials
def graphene_band_structure(t: float = 2.7, a: float = 2.46) -> BandStructureCalculator:
    """
    Create band structure calculator for graphene.
    Parameters
    ----------
    t : float, default 2.7
        Hopping parameter in eV
    a : float, default 2.46
        Lattice constant in Angstrom
    Returns
    -------
    BandStructureCalculator
        Graphene band structure calculator
    """
    # Graphene lattice vectors
    a1 = a * np.array([1.0, 0.0])
    a2 = a * np.array([0.5, np.sqrt(3)/2])
    lattice_vectors = np.array([a1, a2])
    # Basis positions (A and B sublattices)
    basis_positions = np.array([
        [0.0, 0.0],                    # A sublattice
        [a/3, a/(3*np.sqrt(3))]        # B sublattice
    ])
    # Hopping parameters (nearest neighbor only)
    hopping_parameters = {
        (0, 1): t,  # A to B
        (1, 0): t   # B to A
    }
    return BandStructureCalculator(lattice_vectors, basis_positions, hopping_parameters)
def calculate_bands(calculator: BandStructureCalculator,
                   crystal_system: str,
                   n_points: int = 100) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Calculate band structure along high-symmetry path.
    Parameters
    ----------
    calculator : BandStructureCalculator
        Band structure calculator
    crystal_system : str
        Crystal system for high-symmetry path
    n_points : int, default 100
        Number of k-points
    Returns
    -------
    energies : ndarray
        Band energies
    k_path : ndarray
        k-point path
    labels : list
        High-symmetry labels
    k_distances : ndarray
        Distances along path
    """
    k_path, labels, k_distances = calculator.generate_high_symmetry_path(crystal_system, n_points)
    energies, _ = calculator.calculate_bands(k_path)
    return energies, k_path, labels, k_distances
def high_symmetry_path(crystal_system: str, n_points: int = 100) -> Tuple[np.ndarray, List[str]]:
    """
    Generate high-symmetry k-point path for given crystal system.
    Parameters
    ----------
    crystal_system : str
        Crystal system
    n_points : int, default 100
        Number of points
    Returns
    -------
    k_path : ndarray
        k-point path
    labels : list
        Labels for high-symmetry points
    """
    # Create temporary calculator to generate path
    temp_calc = BandStructureCalculator(
        np.eye(2), np.array([[0, 0]]), {(0, 0): 0}
    )
    k_path, labels, _ = temp_calc.generate_high_symmetry_path(crystal_system, n_points)
    return k_path, labels
def plot_band_structure(energies: np.ndarray,
                       k_distances: np.ndarray,
                       labels: Optional[List[str]] = None,
                       fermi_energy: Optional[float] = None,
                       output_dir: Optional[Path] = None) -> plt.Figure:
    """
    Plot band structure with Berkeley styling.
    Parameters
    ----------
    energies : ndarray
        Band energies
    k_distances : ndarray
        k-path distances
    labels : list, optional
        High-symmetry point labels
    fermi_energy : float, optional
        Fermi energy
    output_dir : Path, optional
        Output directory
    Returns
    -------
    Figure
        Matplotlib figure
    """
    berkeley_plot = BerkeleyPlot(figsize=(10, 8))
    fig, ax = berkeley_plot.create_figure()
    # Plot bands
    n_bands = energies.shape[1]
    for band in range(n_bands):
        ax.plot(k_distances, energies[:, band],
               color=BERKELEY_BLUE, linewidth=2, alpha=0.8)
    if fermi_energy is not None:
        ax.axhline(y=fermi_energy, color=CALIFORNIA_GOLD,
                  linestyle='--', linewidth=2, label=f'E_F = {fermi_energy:.2f} eV')
        ax.legend()
    ax.set_xlabel('k-path')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('Electronic Band Structure', fontweight='bold', color=BERKELEY_BLUE)
    ax.grid(True, alpha=0.3)
    if output_dir:
        berkeley_plot.save_figure(output_dir / "band_structure.png")
    return fig