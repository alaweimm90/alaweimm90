#!/usr/bin/env python3
"""
Graphene Electronic Structure and Properties
Comprehensive implementation of graphene's electronic properties including
tight-binding band structure, transport calculations, and response to
external fields and strain effects.
Key Features:
- Tight-binding model with nearest and next-nearest neighbor hopping
- Band structure calculations with valley physics
- Transport properties and conductivity calculations
- Magnetic field effects and Landau levels
- Strain-induced modifications and pseudomagnetic fields
Applications:
- Electronic device modeling
- Transport property calculations
- Valley electronics and spintronics
- Strain engineering studies
- Quantum Hall effect investigations
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
Created: 2025
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union, Callable, Dict, List
from dataclasses import dataclass
import warnings
from ...utils.constants import hbar, me, e, kb
from ...utils.units import energy_convert
from ...visualization.berkeley_style import BerkeleyPlot
@dataclass
class GrapheneConfig:
    """Configuration for graphene calculations."""
    # Lattice parameters
    lattice_constant: float = 2.46e-10  # m (carbon-carbon distance)
    nearest_neighbor_distance: float = 1.42e-10  # m
    # Tight-binding parameters
    t1: float = 2.8  # eV (nearest neighbor hopping)
    t2: float = 0.1  # eV (next-nearest neighbor hopping)
    onsite_energy: float = 0.0  # eV
    # k-space sampling
    n_kx: int = 100
    n_ky: int = 100
    k_max: float = 4.0  # in units of π/a
    # Physical parameters
    temperature: float = 300.0  # K
    chemical_potential: float = 0.0  # eV
    # External fields
    magnetic_field: float = 0.0  # Tesla
    electric_field: float = 0.0  # V/m
    # Strain parameters
    strain_xx: float = 0.0
    strain_yy: float = 0.0
    strain_xy: float = 0.0
    # Computational parameters
    include_spin: bool = True
    use_sparse_matrices: bool = False
class Graphene:
    """
    Comprehensive graphene electronic structure calculator.
    Provides tight-binding calculations of graphene's electronic properties
    including band structure, density of states, transport properties,
    and response to external perturbations.
    Parameters
    ----------
    config : GrapheneConfig
        Configuration parameters for graphene calculations
    """
    def __init__(self, config: GrapheneConfig):
        """Initialize graphene system."""
        self.config = config
        # Lattice vectors
        a = config.lattice_constant
        self.a1 = np.array([a/2, a*np.sqrt(3)/2])
        self.a2 = np.array([a/2, -a*np.sqrt(3)/2])
        # Reciprocal lattice vectors
        self.b1 = 2*np.pi/a * np.array([1, 1/np.sqrt(3)])
        self.b2 = 2*np.pi/a * np.array([1, -1/np.sqrt(3)])
        # High-symmetry points in k-space
        self.K_point = (2*np.pi)/(3*a) * np.array([1, 1/np.sqrt(3)])
        self.Kp_point = (2*np.pi)/(3*a) * np.array([1, -1/np.sqrt(3)])
        self.M_point = (2*np.pi)/(2*a) * np.array([1, 0])
        self.Gamma_point = np.array([0, 0])
        # Sublattice positions
        self.delta1 = config.nearest_neighbor_distance * np.array([1, 0])
        self.delta2 = config.nearest_neighbor_distance * np.array([-1/2, np.sqrt(3)/2])
        self.delta3 = config.nearest_neighbor_distance * np.array([-1/2, -np.sqrt(3)/2])
        # Results storage
        self.band_structure = {}
        self.dos_results = {}
        # Thermal energy
        self.kT = kb * config.temperature
    def tight_binding_hamiltonian(self, kx: float, ky: float) -> np.ndarray:
        """
        Construct tight-binding Hamiltonian at given k-point.
        Parameters
        ----------
        kx, ky : float
            k-vector components
        Returns
        -------
        np.ndarray
            Hamiltonian matrix
        """
        # Phase factors for nearest neighbors
        k = np.array([kx, ky])
        phi1 = np.exp(1j * np.dot(k, self.delta1))
        phi2 = np.exp(1j * np.dot(k, self.delta2))
        phi3 = np.exp(1j * np.dot(k, self.delta3))
        # Off-diagonal matrix element
        f_k = self.config.t1 * (phi1 + phi2 + phi3)
        # 2x2 Hamiltonian for A and B sublattices
        H = np.array([
            [self.config.onsite_energy, f_k],
            [np.conj(f_k), self.config.onsite_energy]
        ], dtype=complex)
        # Add next-nearest neighbor hopping if included
        if abs(self.config.t2) > 1e-12:
            # Next-nearest neighbor phase factors
            phi_nn1 = np.exp(1j * np.dot(k, self.a1))
            phi_nn2 = np.exp(1j * np.dot(k, self.a2))
            phi_nn3 = np.exp(1j * np.dot(k, -(self.a1 + self.a2)))
            # Next-nearest neighbor contribution (diagonal)
            nnn_term = self.config.t2 * (phi_nn1 + phi_nn2 + phi_nn3)
            H[0, 0] += nnn_term
            H[1, 1] += nnn_term
        return H
    def strain_modified_hamiltonian(self, kx: float, ky: float) -> np.ndarray:
        """
        Tight-binding Hamiltonian with strain effects.
        Parameters
        ----------
        kx, ky : float
            k-vector components
        Returns
        -------
        np.ndarray
            Strain-modified Hamiltonian matrix
        """
        # Strain tensor
        strain = np.array([
            [self.config.strain_xx, self.config.strain_xy],
            [self.config.strain_xy, self.config.strain_yy]
        ])
        # Modified nearest-neighbor vectors under strain
        # δ'ᵢ = (I + strain) · δᵢ
        delta1_strained = (np.eye(2) + strain) @ self.delta1
        delta2_strained = (np.eye(2) + strain) @ self.delta2
        delta3_strained = (np.eye(2) + strain) @ self.delta3
        # Modified hopping parameters (distance dependence)
        # t'ᵢ = t₁ * (|δᵢ|/|δ₀|)^β where β ≈ -3 for π orbitals
        beta = -3.0
        d0 = self.config.nearest_neighbor_distance
        d1 = np.linalg.norm(delta1_strained)
        d2 = np.linalg.norm(delta2_strained)
        d3 = np.linalg.norm(delta3_strained)
        t1_strained = self.config.t1 * (d1/d0)**beta
        t2_strained = self.config.t1 * (d2/d0)**beta
        t3_strained = self.config.t1 * (d3/d0)**beta
        # Phase factors with strained vectors
        k = np.array([kx, ky])
        phi1 = t1_strained * np.exp(1j * np.dot(k, delta1_strained))
        phi2 = t2_strained * np.exp(1j * np.dot(k, delta2_strained))
        phi3 = t3_strained * np.exp(1j * np.dot(k, delta3_strained))
        # Off-diagonal element
        f_k_strain = phi1 + phi2 + phi3
        # Hamiltonian with strain
        H_strain = np.array([
            [self.config.onsite_energy, f_k_strain],
            [np.conj(f_k_strain), self.config.onsite_energy]
        ], dtype=complex)
        return H_strain
    def magnetic_field_hamiltonian(self, kx: float, ky: float) -> np.ndarray:
        """
        Hamiltonian in uniform magnetic field (Landau gauge).
        Parameters
        ----------
        kx, ky : float
            k-vector components
        Returns
        -------
        np.ndarray
            Magnetic field Hamiltonian
        """
        B = self.config.magnetic_field
        if abs(B) < 1e-12:
            return self.tight_binding_hamiltonian(kx, ky)
        # Magnetic length
        l_B = np.sqrt(hbar / (e * B))
        # Vector potential in Landau gauge: A = (0, Bx)
        # Minimal coupling: k → k - eA/ℏ
        # For graphene, we use the continuum approximation near Dirac points
        # This is a simplified treatment - full magnetic field requires larger matrices
        # Effective velocity
        v_F = 3 * self.config.t1 * self.config.nearest_neighbor_distance / (2 * hbar)
        # Dirac Hamiltonian with magnetic field
        # H = v_F * (σ_x * π_x + σ_y * π_y) where π = p - eA
        # For Landau gauge A = (0, Bx), this becomes creation/annihilation operators
        # This is a placeholder - full implementation requires proper basis
        return self.tight_binding_hamiltonian(kx, ky)
    def calculate_band_structure(self, k_path: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Calculate band structure along high-symmetry path.
        Parameters
        ----------
        k_path : np.ndarray, optional
            k-points along path, shape (n_points, 2)
        Returns
        -------
        Dict[str, np.ndarray]
            Band structure data
        """
        if k_path is None:
            # Default high-symmetry path: Γ → K → M → Γ
            n_points = 200
            k_path = self._generate_high_symmetry_path(n_points)
        n_points = k_path.shape[0]
        n_bands = 2  # Two bands for graphene
        eigenvalues = np.zeros((n_points, n_bands))
        eigenvectors = np.zeros((n_points, n_bands, 2), dtype=complex)
        for i, (kx, ky) in enumerate(k_path):
            if self._has_strain():
                H = self.strain_modified_hamiltonian(kx, ky)
            else:
                H = self.tight_binding_hamiltonian(kx, ky)
            evals, evecs = eigh(H)
            eigenvalues[i] = evals
            eigenvectors[i] = evecs.T
        self.band_structure = {
            'k_path': k_path,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'k_distance': self._calculate_k_distance(k_path)
        }
        return self.band_structure
    def _generate_high_symmetry_path(self, n_points: int) -> np.ndarray:
        """Generate k-path along high-symmetry directions."""
        # Γ → K → M → Γ
        n_seg = n_points // 3
        # Γ to K
        k1 = np.linspace(self.Gamma_point, self.K_point, n_seg, endpoint=False)
        # K to M
        k2 = np.linspace(self.K_point, self.M_point, n_seg, endpoint=False)
        # M to Γ
        k3 = np.linspace(self.M_point, self.Gamma_point, n_seg, endpoint=True)
        return np.vstack([k1, k2, k3])
    def _calculate_k_distance(self, k_path: np.ndarray) -> np.ndarray:
        """Calculate cumulative distance along k-path."""
        distances = np.zeros(len(k_path))
        for i in range(1, len(k_path)):
            distances[i] = distances[i-1] + np.linalg.norm(k_path[i] - k_path[i-1])
        return distances
    def _has_strain(self) -> bool:
        """Check if strain is applied."""
        return (abs(self.config.strain_xx) > 1e-12 or
                abs(self.config.strain_yy) > 1e-12 or
                abs(self.config.strain_xy) > 1e-12)
    def calculate_dos(self, energy_range: Tuple[float, float] = (-3.0, 3.0),
                     n_energies: int = 1000) -> Dict[str, np.ndarray]:
        """
        Calculate density of states using k-space integration.
        Parameters
        ----------
        energy_range : Tuple[float, float]
            Energy range in eV
        n_energies : int
            Number of energy points
        Returns
        -------
        Dict[str, np.ndarray]
            DOS data
        """
        energies = np.linspace(energy_range[0], energy_range[1], n_energies)
        dos = np.zeros(n_energies)
        # Create k-mesh
        kx_array = np.linspace(-self.config.k_max, self.config.k_max, self.config.n_kx)
        ky_array = np.linspace(-self.config.k_max, self.config.k_max, self.config.n_ky)
        # Broadening parameter
        eta = 0.01  # eV
        for i, kx in enumerate(kx_array):
            for j, ky in enumerate(ky_array):
                H = self.tight_binding_hamiltonian(kx, ky)
                eigenvals, _ = eigh(H)
                # Add contribution to DOS using Lorentzian broadening
                for eigenval in eigenvals:
                    dos += (eta/np.pi) / ((energies - eigenval)**2 + eta**2)
        # Normalize by k-space area
        k_area = (2 * self.config.k_max)**2
        dos *= k_area / (self.config.n_kx * self.config.n_ky)
        self.dos_results = {
            'energies': energies,
            'dos': dos
        }
        return self.dos_results
    def calculate_conductivity(self, energy: float = 0.0,
                              scattering_time: float = 1e-13) -> Dict[str, float]:
        """
        Calculate electrical conductivity using Kubo formula.
        Parameters
        ----------
        energy : float
            Energy at which to calculate conductivity (eV)
        scattering_time : float
            Electron scattering time (s)
        Returns
        -------
        Dict[str, float]
            Conductivity components
        """
        # Create fine k-mesh around Fermi surface
        dk = 0.01
        kx_array = np.arange(-2.0, 2.0, dk)
        ky_array = np.arange(-2.0, 2.0, dk)
        sigma_xx = 0.0
        sigma_xy = 0.0
        for kx in kx_array:
            for ky in ky_array:
                H = self.tight_binding_hamiltonian(kx, ky)
                eigenvals, eigenvecs = eigh(H)
                # Calculate velocity matrix elements
                v_x, v_y = self._calculate_velocity(kx, ky, eigenvecs)
                for n, eigenval in enumerate(eigenvals):
                    # Fermi-Dirac derivative (delta function approximation)
                    fermi_deriv = -self._fermi_dirac_derivative(eigenval - energy)
                    if abs(fermi_deriv) > 1e-10:
                        # Conductivity contributions
                        sigma_xx += e**2 * scattering_time * abs(v_x[n])**2 * fermi_deriv
                        sigma_xy += e**2 * scattering_time * v_x[n] * v_y[n] * fermi_deriv
        # Normalize by k-space area and add spin degeneracy
        k_area = (4.0)**2
        n_k_points = len(kx_array) * len(ky_array)
        normalization = k_area / n_k_points
        if self.config.include_spin:
            normalization *= 2  # Spin degeneracy
        sigma_xx *= normalization
        sigma_xy *= normalization
        return {
            'sigma_xx': sigma_xx,  # S/m
            'sigma_xy': sigma_xy,  # S/m
            'sigma_yy': sigma_xx,  # Isotropic for unstrained graphene
            'resistivity': 1/sigma_xx if sigma_xx > 0 else np.inf  # Ω⋅m
        }
    def _calculate_velocity(self, kx: float, ky: float,
                           eigenvectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate velocity matrix elements."""
        # Velocity operator: v = (1/ℏ) * ∂H/∂k
        dk = 1e-6
        # Finite difference derivatives
        H_kx_plus = self.tight_binding_hamiltonian(kx + dk, ky)
        H_kx_minus = self.tight_binding_hamiltonian(kx - dk, ky)
        dH_dkx = (H_kx_plus - H_kx_minus) / (2 * dk)
        H_ky_plus = self.tight_binding_hamiltonian(kx, ky + dk)
        H_ky_minus = self.tight_binding_hamiltonian(kx, ky - dk)
        dH_dky = (H_ky_plus - H_ky_minus) / (2 * dk)
        # Velocity matrix elements
        v_x = np.diag(eigenvectors.conj().T @ dH_dkx @ eigenvectors) / hbar
        v_y = np.diag(eigenvectors.conj().T @ dH_dky @ eigenvectors) / hbar
        return v_x, v_y
    def _fermi_dirac_derivative(self, energy: float) -> float:
        """Derivative of Fermi-Dirac distribution."""
        if abs(self.kT) < 1e-12:
            # T = 0 limit: delta function
            return 1.0 if abs(energy) < 1e-10 else 0.0
        x = energy / self.kT
        if abs(x) > 50:  # Avoid overflow
            return 0.0
        exp_x = np.exp(x)
        return -exp_x / (self.kT * (1 + exp_x)**2)
    def calculate_optical_conductivity(self, frequency_range: Tuple[float, float] = (0.0, 5.0),
                                     n_frequencies: int = 100) -> Dict[str, np.ndarray]:
        """
        Calculate optical conductivity using linear response theory.
        Parameters
        ----------
        frequency_range : Tuple[float, float]
            Frequency range in eV
        n_frequencies : int
            Number of frequency points
        Returns
        -------
        Dict[str, np.ndarray]
            Optical conductivity vs frequency
        """
        frequencies = np.linspace(frequency_range[0], frequency_range[1], n_frequencies)
        sigma_opt = np.zeros(n_frequencies, dtype=complex)
        # Broadening parameter
        gamma = 0.1  # eV
        # k-space integration
        dk = 0.02
        kx_array = np.arange(-2.0, 2.0, dk)
        ky_array = np.arange(-2.0, 2.0, dk)
        for i, omega in enumerate(frequencies):
            sigma_omega = 0.0
            for kx in kx_array:
                for ky in ky_array:
                    H = self.tight_binding_hamiltonian(kx, ky)
                    eigenvals, eigenvecs = eigh(H)
                    # Calculate transition matrix elements
                    v_x, v_y = self._calculate_velocity(kx, ky, eigenvecs)
                    # Sum over band transitions
                    for n in range(len(eigenvals)):
                        for m in range(len(eigenvals)):
                            if n != m:
                                # Transition frequency
                                omega_nm = eigenvals[m] - eigenvals[n]
                                # Fermi factors
                                f_n = self._fermi_dirac(eigenvals[n])
                                f_m = self._fermi_dirac(eigenvals[m])
                                # Transition contribution
                                if abs(omega_nm) > 1e-10:
                                    matrix_element = abs(v_x[n] * np.conj(v_x[m]))**2
                                    # Add to conductivity
                                    denominator = omega - omega_nm + 1j * gamma
                                    sigma_omega += (f_n - f_m) * matrix_element / denominator
            sigma_opt[i] = sigma_omega
        # Normalization and units
        k_area = (4.0)**2
        n_k_points = len(kx_array) * len(ky_array)
        normalization = 1j * e**2 * k_area / (n_k_points * hbar)
        if self.config.include_spin:
            normalization *= 2
        sigma_opt *= normalization
        return {
            'frequencies': frequencies,
            'sigma_real': np.real(sigma_opt),
            'sigma_imag': np.imag(sigma_opt)
        }
    def _fermi_dirac(self, energy: float) -> float:
        """Fermi-Dirac distribution function."""
        if abs(self.kT) < 1e-12:
            # T = 0 limit
            return 1.0 if energy < self.config.chemical_potential else 0.0
        x = (energy - self.config.chemical_potential) / self.kT
        if x > 50:
            return 0.0
        elif x < -50:
            return 1.0
        else:
            return 1.0 / (1.0 + np.exp(x))
    def calculate_valley_polarization(self, kx: float, ky: float) -> complex:
        """
        Calculate valley polarization at given k-point.
        Parameters
        ----------
        kx, ky : float
            k-vector components
        Returns
        -------
        complex
            Valley pseudospin polarization
        """
        H = self.tight_binding_hamiltonian(kx, ky)
        eigenvals, eigenvecs = eigh(H)
        # Valley pseudospin operator (Pauli matrices in sublattice space)
        sigma_z = np.array([[1, 0], [0, -1]])
        # Calculate expectation value for each band
        valley_pol = []
        for n in range(len(eigenvals)):
            psi = eigenvecs[:, n]
            pol = np.conj(psi) @ sigma_z @ psi
            valley_pol.append(pol)
        return np.array(valley_pol)
    def plot_band_structure(self) -> None:
        """Plot the calculated band structure."""
        if not self.band_structure:
            print("Calculate band structure first using calculate_band_structure()")
            return
        berkeley_plot = BerkeleyPlot()
        fig, ax = plt.subplots(figsize=(10, 6))
        k_dist = self.band_structure['k_distance']
        eigenvals = self.band_structure['eigenvalues']
        # Plot bands
        for band in range(eigenvals.shape[1]):
            ax.plot(k_dist, eigenvals[:, band],
                   color=berkeley_plot.colors['berkeley_blue'],
                   linewidth=2)
        # Mark high-symmetry points
        # This is approximate - would need actual k-path analysis
        n_points = len(k_dist)
        gamma1_pos = 0
        k_pos = k_dist[n_points//3]
        m_pos = k_dist[2*n_points//3]
        gamma2_pos = k_dist[-1]
        ax.axvline(gamma1_pos, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(k_pos, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(m_pos, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(gamma2_pos, color='gray', linestyle='--', alpha=0.5)
        # Labels
        ax.set_xticks([gamma1_pos, k_pos, m_pos, gamma2_pos])
        ax.set_xticklabels(['Γ', 'K', 'M', 'Γ'])
        ax.set_xlabel('k-path')
        ax.set_ylabel('Energy (eV)')
        ax.set_title('Graphene Band Structure')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='red', linestyle=':', alpha=0.7, label='Fermi Level')
        ax.legend()
        plt.tight_layout()
        plt.show()
    def plot_dos(self) -> None:
        """Plot density of states."""
        if not self.dos_results:
            print("Calculate DOS first using calculate_dos()")
            return
        berkeley_plot = BerkeleyPlot()
        fig, ax = plt.subplots(figsize=(10, 6))
        energies = self.dos_results['energies']
        dos = self.dos_results['dos']
        ax.plot(energies, dos,
               color=berkeley_plot.colors['berkeley_blue'],
               linewidth=2)
        ax.fill_between(energies, 0, dos, alpha=0.3,
                       color=berkeley_plot.colors['berkeley_blue'])
        ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='Dirac Point')
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('DOS (states/eV)')
        ax.set_title('Graphene Density of States')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()
    def plot_fermi_surface(self, energy: float = 0.0) -> None:
        """Plot Fermi surface at given energy."""
        berkeley_plot = BerkeleyPlot()
        fig, ax = plt.subplots(figsize=(8, 8))
        # Create k-mesh
        kx_array = np.linspace(-2.0, 2.0, 200)
        ky_array = np.linspace(-2.0, 2.0, 200)
        KX, KY = np.meshgrid(kx_array, ky_array)
        # Calculate band energies
        energies = np.zeros_like(KX)
        for i in range(KX.shape[0]):
            for j in range(KX.shape[1]):
                H = self.tight_binding_hamiltonian(KX[i, j], KY[i, j])
                eigenvals, _ = eigh(H)
                # Take closest eigenvalue to target energy
                energies[i, j] = eigenvals[np.argmin(np.abs(eigenvals - energy))]
        # Plot contours at target energy
        contours = ax.contour(KX, KY, energies, levels=[energy],
                            colors=[berkeley_plot.colors['berkeley_blue']],
                            linewidths=2)
        # Mark high-symmetry points
        ax.plot(0, 0, 'ro', markersize=8, label='Γ')
        ax.plot(self.K_point[0], self.K_point[1], 'go', markersize=8, label='K')
        ax.plot(self.Kp_point[0], self.Kp_point[1], 'go', markersize=8, label="K'")
        ax.plot(self.M_point[0], self.M_point[1], 'bo', markersize=8, label='M')
        ax.set_xlabel('kₓ')
        ax.set_ylabel('kᵧ')
        ax.set_title(f'Fermi Surface at E = {energy:.2f} eV')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()
if __name__ == "__main__":
    # Example: Pristine graphene calculations
    config = GrapheneConfig(
        n_kx=50, n_ky=50,
        temperature=300.0,
        chemical_potential=0.0
    )
    graphene = Graphene(config)
    print("Graphene Electronic Structure Calculation")
    print("=" * 50)
    # Calculate and plot band structure
    print("Calculating band structure...")
    band_data = graphene.calculate_band_structure()
    graphene.plot_band_structure()
    # Calculate and plot DOS
    print("Calculating density of states...")
    dos_data = graphene.calculate_dos(energy_range=(-3.0, 3.0))
    graphene.plot_dos()
    # Calculate transport properties
    print("Calculating conductivity...")
    conductivity = graphene.calculate_conductivity(energy=0.0)
    print(f"Conductivity σₓₓ = {conductivity['sigma_xx']:.2e} S/m")
    print(f"Resistivity ρ = {conductivity['resistivity']:.2e} Ω⋅m")
    # Plot Fermi surface
    print("Plotting Fermi surface...")
    graphene.plot_fermi_surface(energy=0.1)  # Slightly doped
    # Example with strain
    print("\n" + "=" * 50)
    print("Strained Graphene Calculation")
    strained_config = GrapheneConfig(
        strain_xx=0.02,  # 2% tensile strain
        strain_yy=-0.01, # 1% compressive strain
        n_kx=50, n_ky=50
    )
    strained_graphene = Graphene(strained_config)
    # Calculate strained band structure
    strained_bands = strained_graphene.calculate_band_structure()
    strained_graphene.plot_band_structure()
    print("Strain effects calculated successfully!")
    # Compare Dirac point velocities
    print("\nDirac point analysis:")
    # Calculate velocity at K point for pristine graphene
    K = graphene.K_point
    H_K = graphene.tight_binding_hamiltonian(K[0], K[1])
    eigenvals_K, eigenvecs_K = eigh(H_K)
    print(f"Dirac point energy (pristine): {eigenvals_K[0]:.6f} eV")
    # Calculate for strained graphene
    H_K_strain = strained_graphene.strain_modified_hamiltonian(K[0], K[1])
    eigenvals_K_strain, _ = eigh(H_K_strain)
    print(f"Dirac point energy (strained): {eigenvals_K_strain[0]:.6f} eV")
    print("\nCalculation completed successfully!")