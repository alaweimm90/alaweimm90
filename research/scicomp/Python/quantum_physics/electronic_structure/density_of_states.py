#!/usr/bin/env python3
"""
Electronic Density of States Calculations
Comprehensive implementation of density of states (DOS) calculations for
various electronic systems including bulk materials, surfaces, and
nanostructures. Supports analytical and numerical methods with advanced
analysis tools for electronic properties.
Key Features:
- Analytical DOS for free electron gas, 1D/2D/3D systems
- Numerical DOS from band structure calculations
- Local density of states (LDOS) for surfaces and interfaces
- Joint density of states for optical transitions
- Temperature-dependent electronic properties
Applications:
- Electronic structure analysis of materials
- Photoemission spectroscopy interpretation
- Optical and transport property calculations
- Surface and interface physics
- Thermoelectric property prediction
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
Created: 2025
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
import numpy as np
from scipy.integrate import quad, trapz
from scipy.interpolate import griddata, interp1d
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union, Callable, Dict, List
from dataclasses import dataclass
import warnings
from ...utils.constants import hbar, me, kb, e
from ...utils.units import energy_convert
from ...visualization.berkeley_style import BerkeleyPlot
@dataclass
class DOSConfig:
    """Configuration for density of states calculations."""
    energy_range: Tuple[float, float] = (-5.0, 5.0)  # eV
    n_energies: int = 1000
    temperature: float = 300.0  # K
    broadening: float = 0.1  # eV (for numerical DOS)
    fermi_level: float = 0.0  # eV
    # System parameters
    effective_mass: float = me
    lattice_constant: float = 5.0e-10  # m
    # k-space sampling
    n_kpoints: int = 100
    dimension: int = 3  # Dimensionality (1D, 2D, or 3D)
class DensityOfStates:
    """
    Comprehensive density of states calculation toolkit.
    Provides analytical and numerical methods for calculating electronic
    density of states in various material systems with advanced analysis
    capabilities for transport and optical properties.
    Parameters
    ----------
    config : DOSConfig
        Configuration parameters for DOS calculations
    """
    def __init__(self, config: DOSConfig):
        """Initialize density of states calculator."""
        self.config = config
        # Setup energy grid
        self.energies = np.linspace(config.energy_range[0],
                                  config.energy_range[1],
                                  config.n_energies)
        # Convert energies to SI units for calculations
        self.energies_SI = self.energies * e
        # Results storage
        self.dos_results = {}
        self.ldos_results = {}
        # Thermal energy
        self.kT = kb * config.temperature
    def free_electron_dos_3d(self, volume: float = 1.0) -> np.ndarray:
        """
        Analytical DOS for 3D free electron gas.
        Parameters
        ----------
        volume : float
            System volume in m³
        Returns
        -------
        np.ndarray
            Density of states vs energy
        """
        # 3D free electron DOS: g(E) = (1/2π²) * (2m/ℏ²)^(3/2) * √E * V
        dos = np.zeros_like(self.energies)
        # Only positive energies are physical for free electrons
        positive_mask = self.energies > 0
        E_pos = self.energies_SI[positive_mask]
        prefactor = (1 / (2 * np.pi**2)) * ((2 * self.config.effective_mass) / hbar**2)**(3/2) * volume
        dos[positive_mask] = prefactor * np.sqrt(E_pos) / e  # Convert back to eV⁻¹
        return dos
    def free_electron_dos_2d(self, area: float = 1.0) -> np.ndarray:
        """
        Analytical DOS for 2D free electron gas.
        Parameters
        ----------
        area : float
            System area in m²
        Returns
        -------
        np.ndarray
            Density of states vs energy
        """
        # 2D free electron DOS: g(E) = (m/πℏ²) * A (constant)
        dos = np.zeros_like(self.energies)
        positive_mask = self.energies > 0
        dos_value = (self.config.effective_mass / (np.pi * hbar**2)) * area / e
        dos[positive_mask] = dos_value
        return dos
    def free_electron_dos_1d(self, length: float = 1.0) -> np.ndarray:
        """
        Analytical DOS for 1D free electron gas.
        Parameters
        ----------
        length : float
            System length in m
        Returns
        -------
        np.ndarray
            Density of states vs energy
        """
        # 1D free electron DOS: g(E) = (1/π) * √(2m/ℏ²) * L / √E
        dos = np.zeros_like(self.energies)
        positive_mask = self.energies > 0
        E_pos = self.energies_SI[positive_mask]
        prefactor = (1 / np.pi) * np.sqrt(2 * self.config.effective_mass / hbar**2) * length
        dos[positive_mask] = prefactor / np.sqrt(E_pos) / e
        return dos
    def parabolic_band_dos(self, band_bottom: float, effective_mass: float,
                          dimension: int = 3, degeneracy: int = 1) -> np.ndarray:
        """
        DOS for parabolic band with arbitrary effective mass.
        Parameters
        ----------
        band_bottom : float
            Band bottom energy in eV
        effective_mass : float
            Effective mass in units of electron mass
        dimension : int
            Dimensionality (1, 2, or 3)
        degeneracy : int
            Band degeneracy factor
        Returns
        -------
        np.ndarray
            Density of states vs energy
        """
        dos = np.zeros_like(self.energies)
        # Energy relative to band bottom
        E_rel = self.energies - band_bottom
        valid_mask = E_rel > 0
        E_rel_SI = E_rel[valid_mask] * e
        m_eff = effective_mass * me
        if dimension == 3:
            # 3D parabolic band
            prefactor = (1 / (2 * np.pi**2)) * ((2 * m_eff) / hbar**2)**(3/2) * degeneracy
            dos[valid_mask] = prefactor * np.sqrt(E_rel_SI) / e
        elif dimension == 2:
            # 2D parabolic band
            dos_value = (m_eff / (np.pi * hbar**2)) * degeneracy / e
            dos[valid_mask] = dos_value
        elif dimension == 1:
            # 1D parabolic band
            prefactor = (1 / np.pi) * np.sqrt(2 * m_eff / hbar**2) * degeneracy
            dos[valid_mask] = prefactor / np.sqrt(E_rel_SI) / e
        return dos
    def linear_band_dos(self, dirac_point: float, fermi_velocity: float,
                       dimension: int = 2) -> np.ndarray:
        """
        DOS for linear (Dirac) band dispersion.
        Parameters
        ----------
        dirac_point : float
            Dirac point energy in eV
        fermi_velocity : float
            Fermi velocity in m/s
        dimension : int
            Dimensionality (1 or 2)
        Returns
        -------
        np.ndarray
            Density of states vs energy
        """
        dos = np.zeros_like(self.energies)
        E_rel = np.abs(self.energies - dirac_point)
        E_rel_SI = E_rel * e
        if dimension == 2:
            # 2D Dirac cone (graphene-like)
            prefactor = 2 / (np.pi * (hbar * fermi_velocity)**2)
            dos = prefactor * E_rel_SI / e
        elif dimension == 1:
            # 1D linear dispersion
            prefactor = 1 / (np.pi * hbar * fermi_velocity)
            dos = prefactor * np.ones_like(self.energies) / e
        return dos
    def numerical_dos_from_bands(self, k_points: np.ndarray,
                                band_energies: np.ndarray,
                                weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate DOS numerically from band structure data.
        Parameters
        ----------
        k_points : np.ndarray
            k-point coordinates, shape (n_kpoints, dimension)
        band_energies : np.ndarray
            Band energies in eV, shape (n_kpoints, n_bands)
        weights : np.ndarray, optional
            k-point weights for integration
        Returns
        -------
        np.ndarray
            Density of states vs energy
        """
        if weights is None:
            weights = np.ones(len(k_points)) / len(k_points)
        dos = np.zeros_like(self.energies)
        broadening_SI = self.config.broadening * e
        # Sum over all k-points and bands
        for i, energy_band in enumerate(band_energies.T):  # Loop over bands
            for j, (k_point, energy) in enumerate(zip(k_points, energy_band)):
                weight = weights[j]
                # Add Gaussian broadening around each eigenvalue
                energy_SI = energy * e
                gaussian = np.exp(-((self.energies_SI - energy_SI)**2) / (2 * broadening_SI**2))
                gaussian /= np.sqrt(2 * np.pi) * broadening_SI / e  # Normalize
                dos += weight * gaussian
        return dos
    def local_dos_surface(self, surface_potential: Callable[[float], float],
                         z_positions: np.ndarray) -> np.ndarray:
        """
        Calculate local density of states near a surface.
        Parameters
        ----------
        surface_potential : callable
            Surface potential V(z) in eV
        z_positions : np.ndarray
            z-coordinates from surface in m
        Returns
        -------
        np.ndarray
            LDOS vs energy and position, shape (n_energies, n_positions)
        """
        ldos = np.zeros((len(self.energies), len(z_positions)))
        for i, E in enumerate(self.energies):
            for j, z in enumerate(z_positions):
                # Simplified LDOS calculation using WKB approximation
                V_z = surface_potential(z)
                if E > V_z:
                    # Propagating state
                    k_z = np.sqrt(2 * self.config.effective_mass * (E - V_z) * e) / hbar
                    # Standing wave pattern from surface reflection
                    ldos[i, j] = 1.0 + np.cos(2 * k_z * z)**2
                else:
                    # Evanescent state
                    kappa = np.sqrt(2 * self.config.effective_mass * (V_z - E) * e) / hbar
                    ldos[i, j] = np.exp(-2 * kappa * z)
        return ldos
    def joint_dos_optical(self, valence_bands: np.ndarray,
                         conduction_bands: np.ndarray,
                         k_points: np.ndarray) -> np.ndarray:
        """
        Calculate joint density of states for optical transitions.
        Parameters
        ----------
        valence_bands : np.ndarray
            Valence band energies, shape (n_kpoints, n_v_bands)
        conduction_bands : np.ndarray
            Conduction band energies, shape (n_kpoints, n_c_bands)
        k_points : np.ndarray
            k-point coordinates
        Returns
        -------
        np.ndarray
            Joint DOS vs photon energy
        """
        jdos = np.zeros_like(self.energies)
        broadening_SI = self.config.broadening * e
        # Calculate all possible transitions
        for v_band in valence_bands.T:
            for c_band in conduction_bands.T:
                transition_energies = c_band - v_band  # eV
                for trans_E in transition_energies:
                    if trans_E > 0:  # Only upward transitions
                        trans_E_SI = trans_E * e
                        # Add broadened contribution
                        gaussian = np.exp(-((self.energies_SI - trans_E_SI)**2) / (2 * broadening_SI**2))
                        gaussian /= np.sqrt(2 * np.pi) * broadening_SI / e
                        jdos += gaussian / len(k_points)
        return jdos
    def fermi_dirac(self, energy: Union[float, np.ndarray],
                   fermi_level: Optional[float] = None) -> Union[float, np.ndarray]:
        """
        Fermi-Dirac distribution function.
        Parameters
        ----------
        energy : float or array
            Energy in eV
        fermi_level : float, optional
            Fermi level in eV
        Returns
        -------
        float or array
            Fermi-Dirac occupation probability
        """
        if fermi_level is None:
            fermi_level = self.config.fermi_level
        return 1.0 / (1.0 + np.exp((energy - fermi_level) * e / self.kT))
    def electron_density(self, dos: np.ndarray,
                        fermi_level: Optional[float] = None) -> float:
        """
        Calculate electron density from DOS.
        Parameters
        ----------
        dos : np.ndarray
            Density of states vs energy
        fermi_level : float, optional
            Fermi level in eV
        Returns
        -------
        float
            Electron density in m⁻³
        """
        if fermi_level is None:
            fermi_level = self.config.fermi_level
        occupation = self.fermi_dirac(self.energies, fermi_level)
        integrand = dos * occupation
        # Integrate over energy
        n_electrons = trapz(integrand, self.energies)
        return n_electrons
    def electronic_heat_capacity(self, dos: np.ndarray,
                                fermi_level: Optional[float] = None) -> float:
        """
        Calculate electronic heat capacity.
        Parameters
        ----------
        dos : np.ndarray
            Density of states vs energy
        fermi_level : float, optional
            Fermi level in eV
        Returns
        -------
        float
            Electronic heat capacity in J/(K·m³)
        """
        if fermi_level is None:
            fermi_level = self.config.fermi_level
        # Derivative of Fermi-Dirac function
        f = self.fermi_dirac(self.energies, fermi_level)
        df_dE = f * (1 - f) * e / self.kT
        # Heat capacity integrand
        integrand = dos * (self.energies - fermi_level)**2 * e**2 * df_dE / self.kT
        C_el = trapz(integrand, self.energies)
        return C_el
    def seebeck_coefficient(self, dos: np.ndarray, conductivity_dos: np.ndarray,
                           fermi_level: Optional[float] = None) -> float:
        """
        Calculate Seebeck coefficient using Mott's formula.
        Parameters
        ----------
        dos : np.ndarray
            Density of states vs energy
        conductivity_dos : np.ndarray
            Conductivity density of states
        fermi_level : float, optional
            Fermi level in eV
        Returns
        -------
        float
            Seebeck coefficient in V/K
        """
        if fermi_level is None:
            fermi_level = self.config.fermi_level
        # Find Fermi level index
        fermi_idx = np.argmin(np.abs(self.energies - fermi_level))
        # Derivatives at Fermi level (using finite differences)
        if 1 <= fermi_idx <= len(self.energies) - 2:
            dE = self.energies[1] - self.energies[0]
            # Conductivity derivative
            d_sigma_dE = (conductivity_dos[fermi_idx + 1] - conductivity_dos[fermi_idx - 1]) / (2 * dE)
            # Seebeck coefficient (Mott's formula)
            sigma_f = conductivity_dos[fermi_idx]
            if sigma_f > 0:
                S = -(np.pi**2 / 3) * (kb**2 * self.config.temperature / e) * (d_sigma_dE / sigma_f)
            else:
                S = 0.0
        else:
            S = 0.0
        return S
    def plot_dos(self, dos_data: Dict[str, np.ndarray],
                fermi_level: Optional[float] = None) -> None:
        """
        Plot density of states with multiple curves.
        Parameters
        ----------
        dos_data : dict
            Dictionary with {label: DOS array} pairs
        fermi_level : float, optional
            Fermi level to mark on plot
        """
        berkeley_plot = BerkeleyPlot()
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = [berkeley_plot.colors['berkeley_blue'],
                 berkeley_plot.colors['california_gold'],
                 berkeley_plot.colors['green_dark'],
                 berkeley_plot.colors['rose_dark'],
                 berkeley_plot.colors['purple_dark']]
        for i, (label, dos) in enumerate(dos_data.items()):
            color = colors[i % len(colors)]
            ax.plot(self.energies, dos, color=color, linewidth=2, label=label)
        # Mark Fermi level
        if fermi_level is not None:
            ax.axvline(fermi_level, color='red', linestyle='--',
                      alpha=0.7, label=f'E_F = {fermi_level:.2f} eV')
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('DOS (states/eV)')
        ax.set_title('Electronic Density of States')
        ax.grid(True, alpha=0.3)
        ax.legend()
        # Fill occupied states if Fermi level is given
        if fermi_level is not None:
            for label, dos in dos_data.items():
                mask = self.energies <= fermi_level
                ax.fill_between(self.energies[mask], 0, dos[mask],
                               alpha=0.2, label=f'{label} (occupied)')
        plt.tight_layout()
        plt.show()
    def plot_ldos_surface(self, ldos_data: np.ndarray, z_positions: np.ndarray) -> None:
        """
        Plot local density of states as 2D map.
        Parameters
        ----------
        ldos_data : np.ndarray
            LDOS data, shape (n_energies, n_positions)
        z_positions : np.ndarray
            Position coordinates
        """
        berkeley_plot = BerkeleyPlot()
        fig, ax = plt.subplots(figsize=(10, 8))
        # Create meshgrid for plotting
        Z, E = np.meshgrid(z_positions * 1e9, self.energies)
        # Plot as contour map
        contour = ax.contourf(Z, E, ldos_data, levels=50, cmap='RdYlBu_r')
        ax.set_xlabel('Distance from Surface (nm)')
        ax.set_ylabel('Energy (eV)')
        ax.set_title('Local Density of States')
        # Add colorbar
        cbar = plt.colorbar(contour)
        cbar.set_label('LDOS (arb. units)')
        # Mark Fermi level
        ax.axhline(self.config.fermi_level, color='white',
                  linestyle='--', linewidth=2, label='Fermi Level')
        ax.legend()
        plt.tight_layout()
        plt.show()
    def plot_transport_properties(self, dos: np.ndarray) -> None:
        """
        Plot temperature-dependent transport properties.
        Parameters
        ----------
        dos : np.ndarray
            Density of states vs energy
        """
        berkeley_plot = BerkeleyPlot()
        # Temperature range
        temperatures = np.linspace(100, 800, 50)
        heat_capacities = []
        original_temp = self.config.temperature
        for T in temperatures:
            self.config.temperature = T
            self.kT = kb * T
            C_el = self.electronic_heat_capacity(dos)
            heat_capacities.append(C_el)
        # Restore original temperature
        self.config.temperature = original_temp
        self.kT = kb * original_temp
        # Plot heat capacity
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(temperatures, heat_capacities,
               color=berkeley_plot.colors['berkeley_blue'],
               linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('Electronic Heat Capacity (J/(K·m³))')
        ax.set_title('Temperature-Dependent Electronic Heat Capacity')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
if __name__ == "__main__":
    # Example: Compare DOS for different dimensionalities
    config = DOSConfig(
        energy_range=(-2.0, 4.0),
        n_energies=500,
        fermi_level=1.0
    )
    dos_calc = DensityOfStates(config)
    # Calculate DOS for different systems
    dos_3d = dos_calc.free_electron_dos_3d(volume=1e-24)  # nm³
    dos_2d = dos_calc.free_electron_dos_2d(area=1e-16)    # nm²
    dos_1d = dos_calc.free_electron_dos_1d(length=1e-8)   # nm
    # Parabolic bands with different effective masses
    dos_light = dos_calc.parabolic_band_dos(0.5, 0.1, dimension=3)  # Light holes
    dos_heavy = dos_calc.parabolic_band_dos(0.5, 0.5, dimension=3)  # Heavy holes
    # Linear band (graphene-like)
    dos_linear = dos_calc.linear_band_dos(0.0, 1e6, dimension=2)
    # Plot comparison
    dos_data = {
        '3D Free Electron': dos_3d,
        '2D Free Electron': dos_2d * 1e8,  # Scale for visibility
        '1D Free Electron': dos_1d * 1e16, # Scale for visibility
        'Light Holes (3D)': dos_light,
        'Heavy Holes (3D)': dos_heavy,
        'Linear Band (2D)': dos_linear * 1e-12  # Scale for visibility
    }
    dos_calc.plot_dos(dos_data, fermi_level=1.0)
    # Calculate transport properties
    print("Transport Properties at 300K:")
    n_el = dos_calc.electron_density(dos_3d)
    C_el = dos_calc.electronic_heat_capacity(dos_3d)
    print(f"Electron density: {n_el:.2e} m⁻³")
    print(f"Electronic heat capacity: {C_el:.2e} J/(K·m³)")
    # Plot temperature dependence
    dos_calc.plot_transport_properties(dos_3d)