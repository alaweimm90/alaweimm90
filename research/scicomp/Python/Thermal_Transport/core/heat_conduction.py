"""
Heat conduction and thermal transport modeling.
This module provides tools for thermal transport including:
- Heat equation solvers
- Thermal conductivity models
- Phonon transport
- Thermoelectric effects
"""
import numpy as np
from typing import Union, List, Tuple, Optional, Dict, Callable
from scipy.integrate import solve_ivp, quad
from scipy.linalg import solve
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
class HeatEquation:
    """Heat equation solver for various geometries."""
    def __init__(self, thermal_diffusivity: float = 1e-5):
        """
        Initialize heat equation solver.
        Args:
            thermal_diffusivity: Thermal diffusivity α = k/(ρ·c)
        """
        self.alpha = thermal_diffusivity
    def solve_1d_transient(self, x: np.ndarray, t: np.ndarray,
                          initial_condition: Callable,
                          boundary_conditions: Dict,
                          source_term: Optional[Callable] = None) -> np.ndarray:
        """
        Solve 1D transient heat equation.
        Args:
            x: Spatial grid
            t: Time grid
            initial_condition: Initial temperature function
            boundary_conditions: Boundary condition specification
            source_term: Heat source function Q(x,t)
        Returns:
            Temperature field T(x,t)
        """
        nx = len(x)
        nt = len(t)
        dx = x[1] - x[0]
        dt = t[1] - t[0]
        # Stability check
        r = self.alpha * dt / dx**2
        if r > 0.5:
            print(f"Warning: Stability parameter r = {r:.3f} > 0.5")
        # Initialize temperature field
        T = np.zeros((nt, nx))
        T[0, :] = initial_condition(x)
        # Time stepping (explicit finite difference)
        for n in range(nt - 1):
            for i in range(1, nx - 1):
                # Central difference for second derivative
                d2T_dx2 = (T[n, i+1] - 2*T[n, i] + T[n, i-1]) / dx**2
                # Source term
                source = 0
                if source_term is not None:
                    source = source_term(x[i], t[n])
                # Update temperature
                T[n+1, i] = T[n, i] + self.alpha * dt * d2T_dx2 + dt * source
            # Apply boundary conditions
            self._apply_boundary_conditions_1d(T, n+1, x, t[n+1], boundary_conditions)
        return T
    def solve_1d_steady_state(self, x: np.ndarray, boundary_conditions: Dict,
                             source_term: Optional[Callable] = None,
                             thermal_conductivity: Union[float, Callable] = 1.0) -> np.ndarray:
        """
        Solve 1D steady-state heat equation.
        Args:
            x: Spatial grid
            boundary_conditions: Boundary conditions
            source_term: Heat source Q(x)
            thermal_conductivity: Thermal conductivity k(x)
        Returns:
            Temperature distribution T(x)
        """
        nx = len(x)
        dx = x[1] - x[0]
        # Build coefficient matrix for -d/dx(k dT/dx) = Q
        A = np.zeros((nx, nx))
        b = np.zeros(nx)
        # Interior points
        for i in range(1, nx - 1):
            if callable(thermal_conductivity):
                k_left = thermal_conductivity(x[i] - dx/2)
                k_right = thermal_conductivity(x[i] + dx/2)
            else:
                k_left = k_right = thermal_conductivity
            # Finite difference discretization
            A[i, i-1] = k_left / dx**2
            A[i, i] = -(k_left + k_right) / dx**2
            A[i, i+1] = k_right / dx**2
            # Source term
            if source_term is not None:
                b[i] = -source_term(x[i])
        # Apply boundary conditions
        self._apply_boundary_conditions_steady(A, b, x, boundary_conditions)
        # Solve linear system
        T = solve(A, b)
        return T
    def solve_2d_steady_state(self, x: np.ndarray, y: np.ndarray,
                             boundary_conditions: Dict,
                             source_term: Optional[Callable] = None,
                             thermal_conductivity: float = 1.0) -> np.ndarray:
        """
        Solve 2D steady-state heat equation using finite differences.
        Args:
            x: X-direction grid
            y: Y-direction grid
            boundary_conditions: Boundary conditions
            source_term: Heat source Q(x,y)
            thermal_conductivity: Thermal conductivity
        Returns:
            Temperature field T(x,y)
        """
        nx, ny = len(x), len(y)
        dx, dy = x[1] - x[0], y[1] - y[0]
        # Total number of interior points
        n_interior = (nx - 2) * (ny - 2)
        if n_interior <= 0:
            raise ValueError("Grid too small for interior points")
        # Build sparse matrix system
        A = np.zeros((n_interior, n_interior))
        b = np.zeros(n_interior)
        # Map 2D indices to 1D
        def idx(i, j):
            return (i - 1) * (ny - 2) + (j - 1)
        k = 0
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                # Five-point stencil
                A[k, k] = -2 * thermal_conductivity * (1/dx**2 + 1/dy**2)
                # Neighbors
                if i > 1:
                    A[k, idx(i-1, j)] = thermal_conductivity / dx**2
                if i < nx - 2:
                    A[k, idx(i+1, j)] = thermal_conductivity / dx**2
                if j > 1:
                    A[k, idx(i, j-1)] = thermal_conductivity / dy**2
                if j < ny - 2:
                    A[k, idx(i, j+1)] = thermal_conductivity / dy**2
                # Source term
                if source_term is not None:
                    b[k] = -source_term(x[i], y[j])
                k += 1
        # Apply boundary conditions (simplified Dirichlet)
        # This is a simplified implementation - full boundary conditions
        # would require more complex handling
        # Solve sparse system
        T_interior = solve(A, b)
        # Reconstruct full field
        T = np.zeros((nx, ny))
        k = 0
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                T[i, j] = T_interior[k]
                k += 1
        return T
    def _apply_boundary_conditions_1d(self, T: np.ndarray, n: int, x: np.ndarray,
                                     t: float, bc: Dict):
        """Apply boundary conditions for 1D transient problem."""
        nx = len(x)
        dx = x[1] - x[0]
        # Left boundary
        if 'left' in bc:
            if bc['left']['type'] == 'dirichlet':
                T[n, 0] = bc['left']['value']
            elif bc['left']['type'] == 'neumann':
                # dT/dx = value at x=0
                T[n, 0] = T[n, 1] - bc['left']['value'] * dx
            elif bc['left']['type'] == 'robin':
                # h*(T - T_inf) + k*dT/dx = 0
                h, T_inf, k = bc['left']['h'], bc['left']['T_inf'], bc['left']['k']
                T[n, 0] = (T[n, 1] + h*dx*T_inf/k) / (1 + h*dx/k)
        # Right boundary
        if 'right' in bc:
            if bc['right']['type'] == 'dirichlet':
                T[n, nx-1] = bc['right']['value']
            elif bc['right']['type'] == 'neumann':
                T[n, nx-1] = T[n, nx-2] + bc['right']['value'] * dx
            elif bc['right']['type'] == 'robin':
                h, T_inf, k = bc['right']['h'], bc['right']['T_inf'], bc['right']['k']
                T[n, nx-1] = (T[n, nx-2] + h*dx*T_inf/k) / (1 + h*dx/k)
    def _apply_boundary_conditions_steady(self, A: np.ndarray, b: np.ndarray,
                                         x: np.ndarray, bc: Dict):
        """Apply boundary conditions for steady-state problem."""
        nx = len(x)
        # Left boundary (i=0)
        if 'left' in bc:
            if bc['left']['type'] == 'dirichlet':
                A[0, 0] = 1
                A[0, 1:] = 0
                b[0] = bc['left']['value']
        # Right boundary (i=nx-1)
        if 'right' in bc:
            if bc['right']['type'] == 'dirichlet':
                A[nx-1, nx-1] = 1
                A[nx-1, :nx-1] = 0
                b[nx-1] = bc['right']['value']
class PhononTransport:
    """Phonon transport and Boltzmann transport equation."""
    def __init__(self, debye_temperature: float = 300,
                 sound_velocities: Tuple[float, float, float] = (5000, 3000, 3000)):
        """
        Initialize phonon transport model.
        Args:
            debye_temperature: Debye temperature
            sound_velocities: Sound velocities (vL, vT1, vT2)
        """
        self.T_D = debye_temperature
        self.v_s = sound_velocities
        self.k_B = 1.381e-23  # Boltzmann constant
        self.hbar = 1.055e-34  # Reduced Planck constant
    def debye_frequency(self, density: float, n_atoms: float) -> float:
        """
        Calculate Debye cutoff frequency.
        Args:
            density: Material density
            n_atoms: Number of atoms per unit volume
        Returns:
            Debye frequency
        """
        # Average sound velocity
        v_avg = (1/3 * (1/self.v_s[0]**3 + 1/self.v_s[1]**3 + 1/self.v_s[2]**3))**(-1/3)
        # Debye frequency
        omega_D = v_avg * (6 * np.pi**2 * n_atoms)**(1/3)
        return omega_D
    def phonon_specific_heat(self, temperature: float) -> float:
        """
        Calculate phonon specific heat using Debye model.
        Args:
            temperature: Temperature
        Returns:
            Specific heat per unit volume
        """
        if temperature <= 0:
            return 0
        x_D = self.T_D / temperature
        # Debye function
        def integrand(x):
            return x**4 * np.exp(x) / (np.exp(x) - 1)**2
        integral, _ = quad(integrand, 0, x_D)
        C_V = 9 * self.k_B * (temperature / self.T_D)**3 * integral
        return C_V
    def thermal_conductivity_kinetic(self, temperature: float,
                                   mean_free_path: float,
                                   density: float) -> float:
        """
        Calculate thermal conductivity using kinetic theory.
        Args:
            temperature: Temperature
            mean_free_path: Phonon mean free path
            density: Material density
        Returns:
            Thermal conductivity
        """
        # Specific heat
        C_V = self.phonon_specific_heat(temperature)
        # Average sound velocity
        v_avg = np.mean(self.v_s)
        # Kinetic theory: κ = (1/3) * C_V * v * λ
        kappa = (1/3) * C_V * v_avg * mean_free_path
        return kappa
    def umklapp_scattering_rate(self, temperature: float,
                               gruneisen_parameter: float = 1.5) -> float:
        """
        Calculate Umklapp scattering rate.
        Args:
            temperature: Temperature
            gruneisen_parameter: Grüneisen parameter
        Returns:
            Umklapp scattering rate
        """
        # High-temperature limit
        tau_U_inv = gruneisen_parameter**2 * self.k_B * temperature / (self.hbar * self.v_s[0])
        return tau_U_inv
    def boundary_scattering_rate(self, characteristic_length: float) -> float:
        """
        Calculate boundary scattering rate.
        Args:
            characteristic_length: Characteristic sample dimension
        Returns:
            Boundary scattering rate
        """
        v_avg = np.mean(self.v_s)
        tau_B_inv = v_avg / characteristic_length
        return tau_B_inv
    def solve_bte_relaxation_time(self, temperature: float,
                                 scattering_rates: List[float]) -> float:
        """
        Solve BTE using relaxation time approximation.
        Args:
            temperature: Temperature
            scattering_rates: List of scattering rates
        Returns:
            Thermal conductivity
        """
        # Total scattering rate (Matthiessen's rule)
        tau_total_inv = sum(scattering_rates)
        tau_total = 1 / tau_total_inv
        # Mean free path
        v_avg = np.mean(self.v_s)
        lambda_ph = v_avg * tau_total
        # Thermal conductivity
        kappa = self.thermal_conductivity_kinetic(temperature, lambda_ph, 1.0)
        return kappa
class ThermoelectricEffects:
    """Thermoelectric transport phenomena."""
    def __init__(self):
        """Initialize thermoelectric model."""
        self.e = 1.602e-19  # Elementary charge
        self.k_B = 1.381e-23  # Boltzmann constant
    def seebeck_coefficient(self, carrier_concentration: float,
                          effective_mass: float, temperature: float,
                          scattering_parameter: float = 0.5) -> float:
        """
        Calculate Seebeck coefficient.
        Args:
            carrier_concentration: Carrier concentration
            effective_mass: Effective mass
            temperature: Temperature
            scattering_parameter: Scattering parameter (0.5 for acoustic phonons)
        Returns:
            Seebeck coefficient
        """
        # Fermi integral approximation
        eta = np.log(carrier_concentration * (2 * np.pi * effective_mass * self.k_B * temperature)**(3/2) / 2)
        # Seebeck coefficient
        S = (self.k_B / self.e) * (eta + (5/2 + scattering_parameter))
        return S
    def electrical_conductivity(self, carrier_concentration: float,
                              mobility: float) -> float:
        """
        Calculate electrical conductivity.
        Args:
            carrier_concentration: Carrier concentration
            mobility: Carrier mobility
        Returns:
            Electrical conductivity
        """
        sigma = carrier_concentration * self.e * mobility
        return sigma
    def thermal_conductivity_electronic(self, electrical_conductivity: float,
                                      temperature: float,
                                      lorenz_number: float = 2.44e-8) -> float:
        """
        Calculate electronic thermal conductivity.
        Args:
            electrical_conductivity: Electrical conductivity
            temperature: Temperature
            lorenz_number: Lorenz number (Wiedemann-Franz law)
        Returns:
            Electronic thermal conductivity
        """
        kappa_e = lorenz_number * electrical_conductivity * temperature
        return kappa_e
    def figure_of_merit(self, seebeck: float, electrical_conductivity: float,
                       thermal_conductivity: float, temperature: float) -> float:
        """
        Calculate thermoelectric figure of merit ZT.
        Args:
            seebeck: Seebeck coefficient
            electrical_conductivity: Electrical conductivity
            thermal_conductivity: Total thermal conductivity
            temperature: Temperature
        Returns:
            Figure of merit ZT
        """
        power_factor = seebeck**2 * electrical_conductivity
        ZT = power_factor * temperature / thermal_conductivity
        return ZT
    def peltier_coefficient(self, seebeck: float, temperature: float) -> float:
        """
        Calculate Peltier coefficient using Thomson relation.
        Args:
            seebeck: Seebeck coefficient
            temperature: Temperature
        Returns:
            Peltier coefficient
        """
        pi = seebeck * temperature
        return pi
    def thomson_coefficient(self, seebeck: float, temperature: float) -> float:
        """
        Calculate Thomson coefficient.
        Args:
            seebeck: Seebeck coefficient
            temperature: Temperature
        Returns:
            Thomson coefficient
        """
        # Simplified: β = T * dS/dT (assumes linear S vs T)
        beta = temperature * seebeck / temperature  # dS/dT ≈ S/T for simple model
        return beta
class HeatExchanger:
    """Heat exchanger modeling and design."""
    def __init__(self, exchanger_type: str = 'counterflow'):
        """
        Initialize heat exchanger.
        Args:
            exchanger_type: Type of heat exchanger
        """
        self.type = exchanger_type
    def effectiveness_ntu(self, ntu: float, capacity_ratio: float) -> float:
        """
        Calculate effectiveness using NTU method.
        Args:
            ntu: Number of transfer units
            capacity_ratio: Capacity rate ratio (Cmin/Cmax)
        Returns:
            Heat exchanger effectiveness
        """
        C_r = capacity_ratio
        if self.type == 'counterflow':
            if C_r == 1:
                eff = ntu / (1 + ntu)
            else:
                eff = (1 - np.exp(-ntu * (1 - C_r))) / (1 - C_r * np.exp(-ntu * (1 - C_r)))
        elif self.type == 'parallel':
            eff = (1 - np.exp(-ntu * (1 + C_r))) / (1 + C_r)
        elif self.type == 'crossflow':
            # Simplified crossflow (both fluids unmixed)
            eff = 1 - np.exp(ntu**0.22 / C_r * (np.exp(-C_r * ntu**0.78) - 1))
        else:
            raise ValueError(f"Unknown heat exchanger type: {self.type}")
        return eff
    def heat_transfer_rate(self, effectiveness: float, c_min: float,
                          t_hot_in: float, t_cold_in: float) -> float:
        """
        Calculate heat transfer rate.
        Args:
            effectiveness: Heat exchanger effectiveness
            c_min: Minimum capacity rate
            t_hot_in: Hot fluid inlet temperature
            t_cold_in: Cold fluid inlet temperature
        Returns:
            Heat transfer rate
        """
        q_max = c_min * (t_hot_in - t_cold_in)
        q = effectiveness * q_max
        return q
    def pressure_drop(self, reynolds_number: float, friction_factor: float,
                     length: float, diameter: float, density: float,
                     velocity: float) -> float:
        """
        Calculate pressure drop.
        Args:
            reynolds_number: Reynolds number
            friction_factor: Friction factor
            length: Flow length
            diameter: Hydraulic diameter
            density: Fluid density
            velocity: Flow velocity
        Returns:
            Pressure drop
        """
        delta_p = friction_factor * (length / diameter) * (density * velocity**2) / 2
        return delta_p
class NanoscaleHeatTransfer:
    """Nanoscale and microscale heat transfer phenomena."""
    def __init__(self):
        """Initialize nanoscale heat transfer model."""
        self.sigma_SB = 5.67e-8  # Stefan-Boltzmann constant
        self.h_bar = 1.055e-34  # Reduced Planck constant
        self.k_B = 1.381e-23  # Boltzmann constant
    def ballistic_thermal_conductance(self, temperature: float,
                                    contact_area: float,
                                    transmission_coefficient: float = 1.0) -> float:
        """
        Calculate ballistic thermal conductance.
        Args:
            temperature: Temperature
            contact_area: Contact area
            transmission_coefficient: Transmission probability
        Returns:
            Thermal conductance
        """
        # Quantum of thermal conductance
        g_0 = np.pi * self.k_B**2 * temperature / (3 * self.h_bar)
        # Ballistic conductance
        G = transmission_coefficient * g_0 * contact_area
        return G
    def kapitza_resistance(self, debye_temp_1: float, debye_temp_2: float,
                          temperature: float, interface_area: float) -> float:
        """
        Calculate Kapitza thermal boundary resistance.
        Args:
            debye_temp_1: Debye temperature of material 1
            debye_temp_2: Debye temperature of material 2
            temperature: Temperature
            interface_area: Interface area
        Returns:
            Kapitza resistance
        """
        # Simplified acoustic mismatch model
        theta_avg = (debye_temp_1 + debye_temp_2) / 2
        # Kapitza resistance (simplified)
        R_K = theta_avg / (self.k_B * temperature**3 * interface_area)
        return R_K
    def near_field_radiation(self, temperature_1: float, temperature_2: float,
                           gap_distance: float, area: float,
                           material_properties: Dict) -> float:
        """
        Calculate near-field radiative heat transfer.
        Args:
            temperature_1: Temperature of surface 1
            temperature_2: Temperature of surface 2
            gap_distance: Gap distance
            area: Surface area
            material_properties: Material optical properties
        Returns:
            Radiative heat flux
        """
        # Simplified near-field radiation model
        # This is a very simplified version - full calculation requires
        # integration over frequency and wave vector
        # Enhancement factor due to near-field effects
        wavelength_peak = 2.898e-3 / temperature_1  # Wien's law
        enhancement = (wavelength_peak / gap_distance)**2
        # Far-field limit
        q_ff = self.sigma_SB * area * (temperature_1**4 - temperature_2**4)
        # Near-field enhancement
        q_nf = enhancement * q_ff
        return q_nf
    def phonon_tunneling(self, gap_distance: float, phonon_wavelength: float,
                        transmission_probability: float) -> float:
        """
        Calculate phonon tunneling probability.
        Args:
            gap_distance: Gap distance
            phonon_wavelength: Phonon wavelength
            transmission_probability: Base transmission probability
        Returns:
            Tunneling-enhanced transmission
        """
        # Tunneling enhancement factor
        if gap_distance < phonon_wavelength:
            enhancement = np.exp(-2 * gap_distance / phonon_wavelength)
        else:
            enhancement = 0
        T_tunnel = transmission_probability * enhancement
        return T_tunnel