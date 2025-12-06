"""
Spin transport and spintronics device modeling.
This module provides tools for spin transport including:
- Giant magnetoresistance (GMR)
- Tunneling magnetoresistance (TMR)
- Spin diffusion equations
- Spin Hall effects
"""
import numpy as np
from typing import Union, List, Tuple, Optional, Dict
from scipy.integrate import solve_bvp
from scipy.linalg import expm
import matplotlib.pyplot as plt
class SpinDiffusion:
    """Spin diffusion equation solver."""
    def __init__(self, D_up: float, D_down: float, sf_length: float):
        """
        Initialize spin diffusion model.
        Args:
            D_up: Diffusion constant for spin-up electrons
            D_down: Diffusion constant for spin-down electrons
            sf_length: Spin flip length
        """
        self.D_up = D_up
        self.D_down = D_down
        self.lambda_sf = sf_length
        self.D_s = (D_up - D_down) / 2  # Spin diffusion constant
        self.D_c = (D_up + D_down) / 2  # Charge diffusion constant
    def solve_1d(self, x: np.ndarray, boundary_conditions: Dict,
                 current_density: float = 0) -> Dict:
        """
        Solve 1D spin diffusion equation.
        Args:
            x: Position array
            boundary_conditions: Boundary conditions
            current_density: Applied current density
        Returns:
            Solution dictionary with charge and spin densities
        """
        def equations(x, y):
            # y[0] = n_c (charge density)
            # y[1] = dn_c/dx
            # y[2] = n_s (spin density)
            # y[3] = dn_s/dx
            dydt = np.zeros_like(y)
            dydt[0] = y[1]  # dn_c/dx
            dydt[1] = 0     # d²n_c/dx² = 0 (steady state, no sources)
            dydt[2] = y[3]  # dn_s/dx
            dydt[3] = -y[2] / self.lambda_sf**2  # d²n_s/dx² = -n_s/λ_sf²
            return dydt
        def bc(ya, yb):
            # Boundary conditions at x[0] and x[-1]
            bc_left = boundary_conditions.get('left', {'n_c': 0, 'n_s': 0})
            bc_right = boundary_conditions.get('right', {'n_c': 0, 'n_s': 0})
            return np.array([
                ya[0] - bc_left.get('n_c', 0),
                ya[2] - bc_left.get('n_s', 0),
                yb[0] - bc_right.get('n_c', 0),
                yb[2] - bc_right.get('n_s', 0)
            ])
        # Initial guess
        y_init = np.zeros((4, len(x)))
        y_init[0] = np.linspace(
            boundary_conditions['left'].get('n_c', 0),
            boundary_conditions['right'].get('n_c', 0),
            len(x)
        )
        y_init[2] = np.linspace(
            boundary_conditions['left'].get('n_s', 0),
            boundary_conditions['right'].get('n_s', 0),
            len(x)
        ) * np.exp(-np.abs(x - x[len(x)//2]) / self.lambda_sf)
        # Solve boundary value problem
        sol = solve_bvp(equations, bc, x, y_init)
        return {
            'position': sol.x,
            'charge_density': sol.y[0],
            'charge_current': -self.D_c * sol.y[1],
            'spin_density': sol.y[2],
            'spin_current': -self.D_s * sol.y[3],
            'success': sol.success
        }
    def spin_injection_efficiency(self, R_contact: float, R_channel: float,
                                 polarization: float) -> float:
        """
        Calculate spin injection efficiency.
        Args:
            R_contact: Contact resistance
            R_channel: Channel resistance
            polarization: Contact polarization
        Returns:
            Injection efficiency
        """
        # Simplified model
        gamma = R_contact / R_channel
        eta = polarization / (1 + gamma * (1 - polarization**2))
        return eta
class MagnetoresistiveDevices:
    """Giant and tunneling magnetoresistance models."""
    def __init__(self, device_type: str = 'gmr'):
        """
        Initialize MR device.
        Args:
            device_type: 'gmr' or 'tmr'
        """
        self.device_type = device_type.lower()
    def gmr_resistance(self, theta: float, R_parallel: float, R_antiparallel: float,
                      **params) -> float:
        """
        Calculate GMR resistance vs angle.
        Args:
            theta: Angle between magnetizations
            R_parallel: Resistance in parallel configuration
            R_antiparallel: Resistance in antiparallel configuration
        Returns:
            Resistance
        """
        # Simple cosine model
        R = R_parallel + (R_antiparallel - R_parallel) * (1 - np.cos(theta)) / 2
        return R
    def tmr_resistance(self, theta: float, R_parallel: float, TMR_ratio: float,
                      **params) -> float:
        """
        Calculate TMR resistance vs angle.
        Args:
            theta: Angle between magnetizations
            R_parallel: Resistance in parallel configuration
            TMR_ratio: TMR ratio (R_AP - R_P) / R_P
        Returns:
            Resistance
        """
        # Jullière model for TMR
        R = R_parallel * (1 + TMR_ratio * (1 - np.cos(theta)) / 2)
        return R
    def switching_field(self, Ms: float, thickness: float, K_eff: float,
                       alpha: float = 1.0) -> float:
        """
        Calculate switching field for free layer.
        Args:
            Ms: Saturation magnetization
            thickness: Free layer thickness
            K_eff: Effective anisotropy
            alpha: Shape factor
        Returns:
            Switching field
        """
        mu0 = 4 * np.pi * 1e-7
        H_switch = 2 * K_eff / (mu0 * Ms) + alpha * Ms * thickness
        return H_switch
    def thermal_stability(self, Ms: float, volume: float, K_eff: float,
                         temperature: float = 300) -> float:
        """
        Calculate thermal stability factor Δ = KV/kT.
        Args:
            Ms: Saturation magnetization
            volume: Magnetic volume
            K_eff: Effective anisotropy
            temperature: Temperature
        Returns:
            Thermal stability factor
        """
        k_B = 1.381e-23  # Boltzmann constant
        Delta = K_eff * volume / (k_B * temperature)
        return Delta
class SpinHallEffect:
    """Spin Hall effect and spin-orbit coupling."""
    def __init__(self, spin_hall_angle: float = 0.1, conductivity: float = 1e7):
        """
        Initialize spin Hall model.
        Args:
            spin_hall_angle: Spin Hall angle θ_SH
            conductivity: Electrical conductivity
        """
        self.theta_SH = spin_hall_angle
        self.sigma = conductivity
        self.e = 1.602e-19
        self.hbar = 1.055e-34
    def spin_current_from_charge(self, j_charge: np.ndarray) -> np.ndarray:
        """
        Calculate spin current from charge current via SHE.
        Args:
            j_charge: Charge current density vector
        Returns:
            Spin current density
        """
        # Spin current j_s = (ℏ/2e) * θ_SH * (σ × j_charge)
        j_spin = (self.hbar / (2 * self.e)) * self.theta_SH * j_charge
        return j_spin
    def spin_hall_torque(self, j_charge: float, thickness: float,
                        m: np.ndarray, damping_like: bool = True) -> np.ndarray:
        """
        Calculate spin Hall torque.
        Args:
            j_charge: Charge current density
            thickness: Heavy metal thickness
            m: Magnetization direction
            damping_like: Include damping-like torque
        Returns:
            Torque vector
        """
        # Spin current density
        j_s = self.theta_SH * j_charge * self.hbar / (2 * self.e)
        # Spin accumulation direction (y-direction for current in x)
        s = np.array([0, 1, 0])
        # Damping-like torque: τ_DL ∝ m × (m × s)
        if damping_like:
            m_cross_s = np.cross(m, s)
            torque = j_s / thickness * np.cross(m, m_cross_s)
        else:
            # Field-like torque: τ_FL ∝ m × s
            torque = j_s / thickness * np.cross(m, s)
        return torque
    def critical_current_sot(self, Ms: float, thickness: float,
                           K_eff: float, alpha: float = 0.01) -> float:
        """
        Calculate critical current for spin-orbit torque switching.
        Args:
            Ms: Saturation magnetization
            thickness: Magnetic layer thickness
            K_eff: Effective anisotropy
            alpha: Gilbert damping
        Returns:
            Critical current density
        """
        mu0 = 4 * np.pi * 1e-7
        # Critical current for SOT switching
        j_c = (2 * self.e * alpha * Ms * thickness * K_eff) / \
              (self.hbar * self.theta_SH * mu0 * Ms)
        return j_c
class RashbaEffect:
    """Rashba spin-orbit coupling effects."""
    def __init__(self, rashba_parameter: float = 1e-11):
        """
        Initialize Rashba model.
        Args:
            rashba_parameter: Rashba parameter α_R (eV·m)
        """
        self.alpha_R = rashba_parameter
        self.hbar = 1.055e-34
        self.m_e = 9.109e-31
    def rashba_field(self, k: np.ndarray) -> np.ndarray:
        """
        Calculate effective Rashba magnetic field.
        Args:
            k: Wave vector
        Returns:
            Effective field
        """
        # Rashba field: B_R = (α_R/g*μ_B) * (z × k)
        z_hat = np.array([0, 0, 1])
        B_rashba = self.alpha_R / self.hbar * np.cross(z_hat, k)
        return B_rashba
    def spin_precession_frequency(self, k: np.ndarray) -> float:
        """
        Calculate spin precession frequency.
        Args:
            k: Wave vector magnitude
        Returns:
            Precession frequency
        """
        # Ω_R = α_R * k / ℏ
        omega = self.alpha_R * k / self.hbar
        return omega
    def current_induced_field(self, current_density: float,
                            electric_field: float) -> np.ndarray:
        """
        Calculate current-induced effective field.
        Args:
            current_density: Current density
            electric_field: Electric field
        Returns:
            Effective field
        """
        # Simplified model for Rashba-induced field
        sigma = 1e7  # Conductivity
        B_eff = self.alpha_R * electric_field / (self.hbar * sigma)
        return np.array([0, B_eff, 0])  # Assuming field in y-direction
class MagnonTransport:
    """Magnon-based spin transport."""
    def __init__(self, magnon_diffusivity: float = 1e-4,
                 magnon_lifetime: float = 1e-9):
        """
        Initialize magnon transport.
        Args:
            magnon_diffusivity: Magnon diffusion constant
            magnon_lifetime: Magnon lifetime
        """
        self.D_m = magnon_diffusivity
        self.tau_m = magnon_lifetime
        self.lambda_m = np.sqrt(self.D_m * self.tau_m)  # Magnon diffusion length
    def magnon_current(self, mu_gradient: np.ndarray, temperature: float) -> np.ndarray:
        """
        Calculate magnon current from chemical potential gradient.
        Args:
            mu_gradient: Magnon chemical potential gradient
            temperature: Temperature
        Returns:
            Magnon current density
        """
        k_B = 1.381e-23
        # Magnon conductivity (simplified)
        sigma_m = self.D_m / (k_B * temperature)
        j_m = -sigma_m * mu_gradient
        return j_m
    def spin_seebeck_coefficient(self, temperature: float,
                               magnetic_moment: float) -> float:
        """
        Calculate spin Seebeck coefficient.
        Args:
            temperature: Temperature
            magnetic_moment: Magnetic moment per unit volume
        Returns:
            Spin Seebeck coefficient
        """
        k_B = 1.381e-23
        mu_B = 5.788e-5  # Bohr magneton (eV/T)
        # Simplified model
        S_s = magnetic_moment * mu_B / (k_B * temperature)
        return S_s
    def magnon_drag_voltage(self, heat_current: float,
                          magnon_phonon_coupling: float) -> float:
        """
        Calculate voltage from magnon drag effect.
        Args:
            heat_current: Phonon heat current
            magnon_phonon_coupling: Magnon-phonon coupling strength
        Returns:
            Induced voltage
        """
        # Simplified magnon drag model
        V_drag = magnon_phonon_coupling * heat_current * self.lambda_m
        return V_drag
class SpinValve:
    """Complete spin valve device modeling."""
    def __init__(self, layers: List[Dict]):
        """
        Initialize spin valve structure.
        Args:
            layers: List of layer dictionaries with properties
        """
        self.layers = layers
        self.n_layers = len(layers)
    def transfer_matrix_method(self, energy: float, k_parallel: float = 0) -> Dict:
        """
        Calculate transmission using transfer matrix method.
        Args:
            energy: Electron energy
            k_parallel: Parallel momentum component
        Returns:
            Transmission coefficients
        """
        # Initialize transfer matrix
        M_total = np.eye(2, dtype=complex)
        for layer in self.layers:
            thickness = layer['thickness']
            potential = layer.get('potential', 0)
            exchange = layer.get('exchange', 0)
            # Wave vectors for up and down spins
            k_up = np.sqrt(2 * 9.109e-31 * (energy + exchange - potential)) / 1.055e-34
            k_down = np.sqrt(2 * 9.109e-31 * (energy - exchange - potential)) / 1.055e-34
            # Layer transfer matrix (simplified)
            M_layer = np.array([
                [np.exp(1j * k_up * thickness), 0],
                [0, np.exp(1j * k_down * thickness)]
            ])
            M_total = M_total @ M_layer
        # Extract transmission coefficients
        T_up = 1 / np.abs(M_total[0, 0])**2
        T_down = 1 / np.abs(M_total[1, 1])**2
        return {'T_up': T_up, 'T_down': T_down, 'TMR': (T_up - T_down) / T_down}
    def iv_characteristics(self, voltages: np.ndarray,
                          temperature: float = 300) -> Dict:
        """
        Calculate I-V characteristics.
        Args:
            voltages: Voltage array
            temperature: Temperature
        Returns:
            Current and conductance
        """
        currents = np.zeros_like(voltages)
        for i, V in enumerate(voltages):
            # Simple tunneling model
            if self.device_type == 'tmr':
                # Simmons model for tunnel junctions
                barrier_height = 1.0  # eV
                barrier_width = 1e-9  # m
                # Tunneling current (simplified)
                I = V * np.exp(-2 * np.sqrt(2 * 9.109e-31 * barrier_height) *
                              barrier_width / 1.055e-34)
                currents[i] = I
            else:
                # Ohmic conduction for GMR
                currents[i] = V / 1000  # Assume 1kΩ resistance
        conductance = np.gradient(currents, voltages)
        return {'voltage': voltages, 'current': currents, 'conductance': conductance}