#!/usr/bin/env python3
"""
Electronic Structure Under Strain - Strain Engineering
Advanced implementation of strain effects on electronic band structures,
including deformation potential theory, piezoresistance, and strain-induced
modifications of electronic properties in semiconductors and 2D materials.
Key Features:
- Deformation potential theory for band edge shifts
- Piezoresistance and strain-dependent transport
- Strain-induced band gap modulation
- Valley physics in strained 2D materials
- Hydrostatic and biaxial strain effects
Applications:
- Flexible electronics and strain sensors
- Band gap engineering in semiconductors
- Valley electronics in transition metal dichalcogenides
- Mechanical property-electronic coupling
- Strain-tunable optoelectronic devices
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
Created: 2025
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union, Callable, Dict, List
from dataclasses import dataclass
import warnings
from ...utils.constants import hbar, me, e, kb
from ...utils.units import energy_convert
from ...visualization.berkeley_style import BerkeleyPlot
@dataclass
class StrainConfig:
    """Configuration for strain engineering calculations."""
    # Strain parameters
    strain_range: Tuple[float, float] = (-0.05, 0.05)  # Strain range
    n_strain_points: int = 100
    # Material parameters
    lattice_constant: float = 5.0e-10  # m
    bulk_modulus: float = 100e9  # Pa
    shear_modulus: float = 40e9   # Pa
    # Electronic parameters
    band_gap: float = 1.0    # eV
    effective_mass_e: float = 0.1 * me  # Electron effective mass
    effective_mass_h: float = 0.5 * me  # Hole effective mass
    # Deformation potentials (eV)
    deformation_potential_ac: float = -5.0  # Conduction band (hydrostatic)
    deformation_potential_av: float = -1.0  # Valence band (hydrostatic)
    deformation_potential_d: float = 15.0   # Shear deformation potential
    # Temperature
    temperature: float = 300.0  # K
class StrainEngineering:
    """
    Comprehensive strain engineering analysis toolkit.
    Provides methods for calculating strain effects on electronic structure,
    including band edge shifts, mobility changes, and strain-dependent
    transport properties in various material systems.
    Parameters
    ----------
    config : StrainConfig
        Configuration parameters for strain calculations
    """
    def __init__(self, config: StrainConfig):
        """Initialize strain engineering system."""
        self.config = config
        # Setup strain grid
        self.strains = np.linspace(config.strain_range[0],
                                 config.strain_range[1],
                                 config.n_strain_points)
        # Results storage
        self.band_gaps = {}
        self.band_edges = {}
        self.effective_masses = {}
        self.mobilities = {}
        # Thermal energy
        self.kT = kb * config.temperature
    def hydrostatic_strain_band_shift(self, strain: float) -> Tuple[float, float]:
        """
        Calculate band edge shifts under hydrostatic strain.
        Uses deformation potential theory:
        ΔE_c = a_c * Tr(ε)
        ΔE_v = a_v * Tr(ε)
        Parameters
        ----------
        strain : float
            Hydrostatic strain component
        Returns
        -------
        Tuple[float, float]
            Conduction and valence band edge shifts (eV)
        """
        # Hydrostatic strain is Tr(ε) = 3 * strain for isotropic case
        hydrostatic_strain = 3 * strain
        # Band edge shifts
        delta_Ec = self.config.deformation_potential_ac * hydrostatic_strain
        delta_Ev = self.config.deformation_potential_av * hydrostatic_strain
        return delta_Ec, delta_Ev
    def biaxial_strain_band_shift(self, strain_xx: float, strain_yy: float,
                                 strain_xy: float = 0.0) -> Tuple[float, float]:
        """
        Calculate band edge shifts under biaxial strain.
        Parameters
        ----------
        strain_xx : float
            Strain component in x-direction
        strain_yy : float
            Strain component in y-direction
        strain_xy : float
            Shear strain component
        Returns
        -------
        Tuple[float, float]
            Conduction and valence band edge shifts (eV)
        """
        # Hydrostatic component
        hydrostatic_strain = strain_xx + strain_yy
        # Shear component
        shear_strain = strain_xx - strain_yy
        # Band edge shifts
        delta_Ec_hydro = self.config.deformation_potential_ac * hydrostatic_strain
        delta_Ev_hydro = self.config.deformation_potential_av * hydrostatic_strain
        # Shear deformation affects valence band more strongly
        delta_Ec_shear = 0.5 * self.config.deformation_potential_d * abs(shear_strain)
        delta_Ev_shear = -0.5 * self.config.deformation_potential_d * abs(shear_strain)
        delta_Ec = delta_Ec_hydro + delta_Ec_shear
        delta_Ev = delta_Ev_hydro + delta_Ev_shear
        return delta_Ec, delta_Ev
    def strain_dependent_band_gap(self, strain_tensor: np.ndarray) -> float:
        """
        Calculate strain-dependent band gap.
        Parameters
        ----------
        strain_tensor : np.ndarray
            3x3 strain tensor
        Returns
        -------
        float
            Modified band gap (eV)
        """
        # Extract strain invariants
        trace_strain = np.trace(strain_tensor)  # Hydrostatic strain
        # Deviatoric strain (shear components)
        deviatoric = strain_tensor - (trace_strain / 3) * np.eye(3)
        shear_magnitude = np.sqrt(0.5 * np.sum(deviatoric**2))
        # Band edge shifts
        delta_Ec = self.config.deformation_potential_ac * trace_strain
        delta_Ev = self.config.deformation_potential_av * trace_strain
        # Additional shear contributions
        delta_gap_shear = self.config.deformation_potential_d * shear_magnitude
        # Modified band gap
        Eg_strained = self.config.band_gap + (delta_Ec - delta_Ev) + delta_gap_shear
        return max(Eg_strained, 0.0)  # Ensure positive band gap
    def strain_dependent_effective_mass(self, strain: float,
                                      mass_type: str = 'electron') -> float:
        """
        Calculate strain-dependent effective mass.
        Parameters
        ----------
        strain : float
            Applied strain
        mass_type : str
            'electron' or 'hole'
        Returns
        -------
        float
            Modified effective mass
        """
        # Empirical strain dependence (simplified model)
        # m*(ε) = m*₀ * (1 + α_m * ε)
        if mass_type == 'electron':
            m0 = self.config.effective_mass_e
            alpha_m = 2.0  # Strain coupling parameter
        else:  # hole
            m0 = self.config.effective_mass_h
            alpha_m = -1.0  # Different sign for holes
        m_strained = m0 * (1 + alpha_m * strain)
        return max(m_strained, 0.1 * me)  # Minimum mass constraint
    def piezoresistance_coefficient(self, strain: float,
                                   carrier_type: str = 'electron') -> float:
        """
        Calculate piezoresistance coefficient.
        π = (1/ρ) * (dρ/dε)
        Parameters
        ----------
        strain : float
            Applied strain
        carrier_type : str
            'electron' or 'hole'
        Returns
        -------
        float
            Piezoresistance coefficient (1/Pa)
        """
        # Get strain-dependent effective mass
        m_strain = self.strain_dependent_effective_mass(strain, carrier_type)
        m_0 = (self.config.effective_mass_e if carrier_type == 'electron'
               else self.config.effective_mass_h)
        # Mobility changes with effective mass as μ ∝ 1/m*
        relative_mass_change = (m_strain - m_0) / m_0
        # Piezoresistance from mobility change
        # ρ ∝ 1/μ ∝ m*, so dρ/ρ = dm*/m*
        # Convert strain to stress using elastic modulus
        stress = self.config.bulk_modulus * strain
        if abs(stress) > 0:
            pi_coefficient = relative_mass_change / stress
        else:
            pi_coefficient = 0.0
        return pi_coefficient
    def valley_splitting_2d(self, strain_xx: float, strain_yy: float) -> Tuple[float, float]:
        """
        Calculate valley splitting in 2D materials under strain.
        For materials like MoS₂, WSe₂ with K and K' valleys.
        Parameters
        ----------
        strain_xx : float
            Strain in x-direction
        strain_yy : float
            Strain in y-direction
        Returns
        -------
        Tuple[float, float]
            Energy shifts for K and K' valleys (eV)
        """
        # Valley coupling parameters (material-specific)
        lambda_K = 0.1   # eV (coupling strength for K valley)
        lambda_Kp = -0.1  # eV (coupling strength for K' valley)
        # Strain-induced valley shifts
        delta_K = lambda_K * (strain_xx - strain_yy)
        delta_Kp = lambda_Kp * (strain_xx - strain_yy)
        return delta_K, delta_Kp
    def strain_induced_polarization(self, strain_tensor: np.ndarray,
                                   piezoelectric_tensor: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate strain-induced electric polarization.
        P_i = d_ijk * σ_jk (piezoelectric effect)
        Parameters
        ----------
        strain_tensor : np.ndarray
            3x3 strain tensor
        piezoelectric_tensor : np.ndarray, optional
            Piezoelectric coefficient tensor
        Returns
        -------
        np.ndarray
            Induced polarization vector (C/m²)
        """
        if piezoelectric_tensor is None:
            # Default piezoelectric coefficients (simplified)
            d = np.zeros((3, 3, 3))
            d[2, 0, 0] = 1e-12  # d₃₁ (C/N)
            d[2, 1, 1] = 1e-12  # d₃₂
            d[2, 2, 2] = 2e-12  # d₃₃
        else:
            d = piezoelectric_tensor
        # Convert strain to stress (simplified)
        stress_tensor = self.config.bulk_modulus * strain_tensor
        # Calculate polarization
        polarization = np.zeros(3)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    polarization[i] += d[i, j, k] * stress_tensor[j, k]
        return polarization
    def optimize_strain_for_band_gap(self, target_gap: float) -> Dict[str, float]:
        """
        Find optimal strain to achieve target band gap.
        Parameters
        ----------
        target_gap : float
            Target band gap in eV
        Returns
        -------
        Dict[str, float]
            Optimal strain parameters
        """
        def objective(strain_params):
            """Objective function to minimize."""
            strain_xx, strain_yy = strain_params
            strain_tensor = np.diag([strain_xx, strain_yy, 0.0])
            current_gap = self.strain_dependent_band_gap(strain_tensor)
            return (current_gap - target_gap)**2
        # Initial guess
        x0 = [0.0, 0.0]
        # Constraints (reasonable strain limits)
        bounds = [(-0.1, 0.1), (-0.1, 0.1)]
        # Optimize
        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        if result.success:
            optimal_strain = {
                'strain_xx': result.x[0],
                'strain_yy': result.x[1],
                'achieved_gap': target_gap,
                'error': result.fun
            }
        else:
            optimal_strain = {
                'strain_xx': 0.0,
                'strain_yy': 0.0,
                'achieved_gap': self.config.band_gap,
                'error': float('inf')
            }
        return optimal_strain
    def thermal_expansion_strain(self, temperature: float,
                               reference_temp: float = 300.0,
                               expansion_coefficient: float = 5e-6) -> float:
        """
        Calculate thermal expansion strain.
        Parameters
        ----------
        temperature : float
            Current temperature (K)
        reference_temp : float
            Reference temperature (K)
        expansion_coefficient : float
            Linear thermal expansion coefficient (1/K)
        Returns
        -------
        float
            Thermal strain
        """
        delta_T = temperature - reference_temp
        thermal_strain = expansion_coefficient * delta_T
        return thermal_strain
    def plot_strain_band_gap_relation(self) -> None:
        """Plot band gap vs strain relationship."""
        berkeley_plot = BerkeleyPlot()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        # Hydrostatic strain effects
        band_gaps_hydro = []
        for strain in self.strains:
            strain_tensor = strain * np.eye(3)  # Isotropic strain
            gap = self.strain_dependent_band_gap(strain_tensor)
            band_gaps_hydro.append(gap)
        ax1.plot(self.strains * 100, band_gaps_hydro,
                color=berkeley_plot.colors['berkeley_blue'],
                linewidth=2, label='Hydrostatic')
        # Biaxial strain effects
        band_gaps_biaxial = []
        for strain in self.strains:
            strain_tensor = np.diag([strain, strain, 0.0])
            gap = self.strain_dependent_band_gap(strain_tensor)
            band_gaps_biaxial.append(gap)
        ax1.plot(self.strains * 100, band_gaps_biaxial,
                color=berkeley_plot.colors['california_gold'],
                linewidth=2, label='Biaxial')
        ax1.set_xlabel('Strain (%)')
        ax1.set_ylabel('Band Gap (eV)')
        ax1.set_title('Strain-Dependent Band Gap')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        # Effective mass vs strain
        masses_e = []
        masses_h = []
        for strain in self.strains:
            m_e = self.strain_dependent_effective_mass(strain, 'electron') / me
            m_h = self.strain_dependent_effective_mass(strain, 'hole') / me
            masses_e.append(m_e)
            masses_h.append(m_h)
        ax2.plot(self.strains * 100, masses_e,
                color=berkeley_plot.colors['green_dark'],
                linewidth=2, label='Electron')
        ax2.plot(self.strains * 100, masses_h,
                color=berkeley_plot.colors['rose_dark'],
                linewidth=2, label='Hole')
        ax2.set_xlabel('Strain (%)')
        ax2.set_ylabel('Effective Mass (m₀)')
        ax2.set_title('Strain-Dependent Effective Mass')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        plt.tight_layout()
        plt.show()
    def plot_piezoresistance(self) -> None:
        """Plot piezoresistance coefficients."""
        berkeley_plot = BerkeleyPlot()
        fig, ax = plt.subplots(figsize=(10, 6))
        pi_electrons = []
        pi_holes = []
        for strain in self.strains:
            pi_e = self.piezoresistance_coefficient(strain, 'electron')
            pi_h = self.piezoresistance_coefficient(strain, 'hole')
            pi_electrons.append(pi_e * 1e9)  # Convert to 1/GPa
            pi_holes.append(pi_h * 1e9)
        ax.plot(self.strains * 100, pi_electrons,
               color=berkeley_plot.colors['berkeley_blue'],
               linewidth=2, label='Electrons')
        ax.plot(self.strains * 100, pi_holes,
               color=berkeley_plot.colors['california_gold'],
               linewidth=2, label='Holes')
        ax.set_xlabel('Strain (%)')
        ax.set_ylabel('Piezoresistance Coefficient (1/GPa)')
        ax.set_title('Piezoresistance vs Strain')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()
    def plot_valley_splitting(self) -> None:
        """Plot valley splitting in 2D materials."""
        berkeley_plot = BerkeleyPlot()
        fig, ax = plt.subplots(figsize=(10, 6))
        # Create strain mesh for 2D plot
        strain_range = np.linspace(-0.03, 0.03, 50)
        strain_xx_mesh, strain_yy_mesh = np.meshgrid(strain_range, strain_range)
        valley_splitting = np.zeros_like(strain_xx_mesh)
        for i in range(len(strain_range)):
            for j in range(len(strain_range)):
                delta_K, delta_Kp = self.valley_splitting_2d(
                    strain_xx_mesh[i, j], strain_yy_mesh[i, j])
                valley_splitting[i, j] = abs(delta_K - delta_Kp)
        contour = ax.contourf(strain_xx_mesh * 100, strain_yy_mesh * 100,
                            valley_splitting * 1000,  # Convert to meV
                            levels=20, cmap='RdYlBu_r')
        ax.set_xlabel('Strain εₓₓ (%)')
        ax.set_ylabel('Strain εᵧᵧ (%)')
        ax.set_title('Valley Splitting in 2D Material')
        # Add colorbar
        cbar = plt.colorbar(contour)
        cbar.set_label('Valley Splitting (meV)')
        plt.tight_layout()
        plt.show()
    def generate_strain_report(self, strain_tensor: np.ndarray) -> Dict[str, float]:
        """
        Generate comprehensive strain analysis report.
        Parameters
        ----------
        strain_tensor : np.ndarray
            3x3 strain tensor to analyze
        Returns
        -------
        Dict[str, float]
            Comprehensive strain analysis results
        """
        # Basic strain invariants
        trace_strain = np.trace(strain_tensor)
        hydrostatic_strain = trace_strain / 3
        # Deviatoric strain
        deviatoric = strain_tensor - hydrostatic_strain * np.eye(3)
        von_mises_strain = np.sqrt(2/3 * np.sum(deviatoric**2))
        # Electronic properties
        band_gap = self.strain_dependent_band_gap(strain_tensor)
        delta_Ec, delta_Ev = self.hydrostatic_strain_band_shift(hydrostatic_strain)
        # Effective masses
        m_e = self.strain_dependent_effective_mass(hydrostatic_strain, 'electron')
        m_h = self.strain_dependent_effective_mass(hydrostatic_strain, 'hole')
        # Piezoresistance
        pi_e = self.piezoresistance_coefficient(hydrostatic_strain, 'electron')
        pi_h = self.piezoresistance_coefficient(hydrostatic_strain, 'hole')
        # Valley effects (if 2D)
        delta_K, delta_Kp = self.valley_splitting_2d(strain_tensor[0, 0],
                                                    strain_tensor[1, 1])
        report = {
            'hydrostatic_strain': hydrostatic_strain,
            'von_mises_strain': von_mises_strain,
            'band_gap_eV': band_gap,
            'conduction_shift_eV': delta_Ec,
            'valence_shift_eV': delta_Ev,
            'electron_mass_ratio': m_e / me,
            'hole_mass_ratio': m_h / me,
            'piezoresistance_electron_GPa': pi_e * 1e9,
            'piezoresistance_hole_GPa': pi_h * 1e9,
            'valley_K_shift_meV': delta_K * 1000,
            'valley_Kp_shift_meV': delta_Kp * 1000,
            'valley_splitting_meV': abs(delta_K - delta_Kp) * 1000
        }
        return report
if __name__ == "__main__":
    # Example: Strain engineering analysis for semiconductor
    config = StrainConfig(
        strain_range=(-0.05, 0.05),
        band_gap=1.5,  # eV
        deformation_potential_ac=-8.0,  # eV
        deformation_potential_av=-1.5,  # eV
        deformation_potential_d=20.0    # eV
    )
    strain_eng = StrainEngineering(config)
    # Plot strain-dependent properties
    strain_eng.plot_strain_band_gap_relation()
    strain_eng.plot_piezoresistance()
    strain_eng.plot_valley_splitting()
    # Find optimal strain for specific band gap
    target_gap = 1.2  # eV
    optimal = strain_eng.optimize_strain_for_band_gap(target_gap)
    print(f"Optimal strain for {target_gap} eV band gap:")
    print(f"εₓₓ = {optimal['strain_xx']:.4f}")
    print(f"εᵧᵧ = {optimal['strain_yy']:.4f}")
    print(f"Achieved gap = {optimal['achieved_gap']:.4f} eV")
    # Generate comprehensive report for biaxial tension
    strain_tensor = np.diag([0.02, 0.02, -0.01])  # 2% biaxial tension
    report = strain_eng.generate_strain_report(strain_tensor)
    print("\nStrain Analysis Report:")
    print("=" * 40)
    for key, value in report.items():
        print(f"{key}: {value:.4f}")
    # Thermal strain analysis
    thermal_strain = strain_eng.thermal_expansion_strain(400.0, 300.0, 5e-6)
    print(f"\nThermal strain at 400K: {thermal_strain:.6f}")
    thermal_tensor = thermal_strain * np.eye(3)
    thermal_gap = strain_eng.strain_dependent_band_gap(thermal_tensor)
    print(f"Band gap at 400K: {thermal_gap:.4f} eV")